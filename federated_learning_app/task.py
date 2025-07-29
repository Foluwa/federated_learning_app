from collections import OrderedDict
from torchvision.transforms import Grayscale
from datetime import datetime
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor, Pad

import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Import the StudentModel from knowledge_distillation
from federated_learning_app.knowledge_distillation import StudentModel

# Replace dataset in load_data
from medmnist import PathMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

# CRITICAL FIX: Use StudentModel instead of original Net
def Net():
    """Return the compressed student model - MUST match server model"""
    return StudentModel(num_classes=9)  # 9 classes for PathMNIST


# Alternative: Keep the original Net class but update it to match StudentModel
class Net_Original(nn.Module):
    """Original model class - UPDATED to match StudentModel architecture"""

    def __init__(self):
        super(Net_Original, self).__init__()
        # UPDATED: Match StudentModel architecture exactly
        self.conv1 = nn.Conv2d(1, 4, 5)  # Changed from 6 to 4 channels
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5)   # Changed from 16 to 8 channels  
        self.fc1 = nn.Linear(8 * 4 * 4, 60)  # Updated dimensions
        self.fc2 = nn.Linear(60, 32)         # Updated dimensions
        self.fc3 = nn.Linear(32, 9)          # Changed from 10 to 9 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def generate_run_name(model_name="CNN", tag="client", round_num=None):
    """Generate custom run name with model, time, and ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    
    if round_num is not None:
        return f"{model_name}_{tag}_R{round_num}_{timestamp}_{short_id}"
    else:
        return f"{model_name}_{tag}_{timestamp}_{short_id}"
    
# Define global variable
fds = None

def get_transforms():
    """Create transform function for PathMNIST"""
    from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale
    
    transform = Compose([
        Grayscale(num_output_channels=1),
        ToTensor(), 
        Normalize((0.5,), (0.5,))
    ])
    
    def apply_transforms(batch):
        # Handle manual transformation for individual images
        transformed_images = []
        for img in batch["image"]:
            # Convert numpy array to PIL Image if needed
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            transformed_images.append(transform(img))
        
        batch["image"] = transformed_images
        return batch
    
    return apply_transforms

def load_data(partition_id: int, num_partitions: int):
    """Load PathMNIST data and partition it for federated learning - FIXED"""
    from medmnist import PathMNIST
    from torch.utils.data import TensorDataset, Subset
    import numpy as np
    
    print(f"📊 Loading PathMNIST data for client {partition_id}/{num_partitions}")
    
    # Load PathMNIST using medmnist library (same as server)
    raw_dataset = PathMNIST(split='train', download=True, as_rgb=True)
    
    # Simple partitioning: each client gets a subset
    total_size = len(raw_dataset)
    samples_per_client = total_size // num_partitions
    start = partition_id * samples_per_client
    end = start + samples_per_client if partition_id < num_partitions - 1 else total_size
    
    # Extract subset
    subset_imgs = raw_dataset.imgs[start:end]
    subset_labels = raw_dataset.labels[start:end].squeeze()
    
    # Convert to grayscale and normalize manually
    processed_imgs = []
    for img in subset_imgs:
        # Convert to grayscale
        if len(img.shape) == 3:
            gray_img = np.dot(img, [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray_img = img
            
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(gray_img).float().unsqueeze(0) / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        processed_imgs.append(img_tensor)
    
    # Create dataset
    images_tensor = torch.stack(processed_imgs)
    labels_tensor = torch.from_numpy(subset_labels).long()
    
    full_dataset = TensorDataset(images_tensor, labels_tensor)
    
    # Split train/test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)
    
    print(f"✅ Client {partition_id}: {len(train_dataset)} train, {len(test_dataset)} test")
    return trainloader, testloader

def train(net, trainloader, epochs, lr, device, round_num=None):
    # Set matplotlib to use non-GUI backend to avoid threading issues
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            # Handle both dictionary format (new) and tuple format (TensorDataset)
            if isinstance(batch, dict):
                images, labels = batch["image"].to(device), batch["label"].to(device)
            else:
                # Tuple format from TensorDataset
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(trainloader))

    # Log training curve
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_losses, label="Train Loss")
        ax.set_title("Client Training Loss Curve (Student Model)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        mlflow.log_figure(fig, "client_train_loss.png")
        plt.close(fig)
    except Exception as e:
        print(f"Training loss plotting failed: {str(e)}")

    return train_losses[-1]

def test(net, testloader, device, tag="client", round_num=None):
    # Set matplotlib to use non-GUI backend to avoid threading issues
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    y_true, y_pred, scores = [], [], []
    loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            # Handle both dictionary format (client) and tuple format (server)
            if isinstance(batch, dict):
                # Dictionary format from HuggingFace datasets
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
            else:
                # Tuple format from TensorDataset
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # Metrics
    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Log to MLflow with custom run name
    custom_run_name = generate_run_name("PathMNIST_Student_CNN", tag, round_num)
    
    with mlflow.start_run(run_name=custom_run_name, nested=True):
        mlflow.set_tag("model_type", tag)
        mlflow.set_tag("model_name", "PathMNIST_Student_CNN")
        mlflow.set_tag("dataset", "PathMNIST")
        mlflow.set_tag("compressed_model", True)
        if round_num is not None:
            mlflow.set_tag("round", round_num)
            
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("loss", loss / len(testloader))

        # Confusion Matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax)
            plt.title(f"Confusion Matrix - {tag} (Student Model)")
            mlflow.log_figure(fig, f"conf_matrix_{tag}.png")
            plt.close(fig)
        except Exception as e:
            print(f"Confusion matrix plotting failed: {str(e)}")

        # ROC Curve (macro-average)
        try:
            # PathMNIST has 9 classes
            y_true_bin = np.eye(9)[y_true]  
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), np.array(scores).ravel())
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label="ROC curve")
            ax.set_title(f"ROC Curve - {tag} (Student Model)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            mlflow.log_figure(fig, f"roc_curve_{tag}.png")
            plt.close(fig)
        except Exception as e:
            print(f"ROC curve plotting failed: {str(e)}")

    return loss / len(testloader), accuracy

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

     
# from collections import OrderedDict
# from torchvision.transforms import Grayscale
# from datetime import datetime
# import uuid
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset, TensorDataset
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import DirichletPartitioner
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, Normalize, ToTensor, Pad

# import mlflow
# import mlflow.pytorch
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
# import matplotlib.pyplot as plt
# import numpy as np

# # Replace dataset in load_data
# from medmnist import PathMNIST
# from torchvision.transforms import Compose, ToTensor, Normalize

# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

# def generate_run_name(model_name="CNN", tag="client", round_num=None):
#     """Generate custom run name with model, time, and ID"""
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     short_id = str(uuid.uuid4())[:8]
    
#     if round_num is not None:
#         return f"{model_name}_{tag}_R{round_num}_{timestamp}_{short_id}"
#     else:
#         return f"{model_name}_{tag}_{timestamp}_{short_id}"
    
# # Define global variable
# fds = None

# # def get_transforms():
# #     from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale
    
# #     transform = Compose([
# #         Grayscale(num_output_channels=1),  # Add this line
# #         ToTensor(), 
# #         Normalize((0.5,), (0.5,))
# #     ])
    
# #     def apply_transforms(batch):
# #         batch["image"] = [transform(img) for img in batch["image"]]
# #         return batch
# #     return apply_transforms

# def get_transforms():
#     """Create transform function for PathMNIST"""
#     from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale
#     from PIL import Image
    
#     transform = Compose([
#         Grayscale(num_output_channels=1),
#         ToTensor(), 
#         Normalize((0.5,), (0.5,))
#     ])
    
#     def apply_transforms(batch):
#         # Handle manual transformation for individual images
#         transformed_images = []
#         for img in batch["image"]:
#             # Convert numpy array to PIL Image if needed
#             if isinstance(img, np.ndarray):
#                 img = Image.fromarray(img)
#             transformed_images.append(transform(img))
        
#         batch["image"] = transformed_images
#         return batch
    
#     return apply_transforms

# # def load_data(partition_id: int, num_partitions: int):
# #     global fds
# #     from flwr_datasets.partitioner import DirichletPartitioner
# #     from flwr_datasets import FederatedDataset

# #     if fds is None:
# #         partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=1.0)
# #         fds = FederatedDataset(dataset="medmnist/pathmnist", partitioners={"train": partitioner})

# #     partition = fds.load_partition(partition_id)
# #     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
# #     partition_train_test = partition_train_test.with_transform(get_transforms())
# #     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
# #     testloader = DataLoader(partition_train_test["test"], batch_size=32)
# #     return trainloader, testloader

# def load_data(partition_id: int, num_partitions: int):
#     """Load PathMNIST data and partition it for federated learning"""
#     from medmnist import PathMNIST
#     from torch.utils.data import TensorDataset, Subset
#     import numpy as np
    
#     print(f"📊 Loading PathMNIST data for client {partition_id}/{num_partitions}")
    
#     # Load PathMNIST using medmnist library (same as server)
#     transform = get_transforms()
    
#     # Load training data
#     train_dataset = PathMNIST(split='train', download=True, as_rgb=True, transform=None)
    
#     # Convert to tensors and apply transforms manually
#     processed_imgs = []
#     labels = []
    
#     for i in range(len(train_dataset.imgs)):
#         img = train_dataset.imgs[i]
#         label = train_dataset.labels[i].item()  # Convert to scalar
        
#         # Apply transforms (grayscale conversion + normalization)
#         img_tensor = transform({'image': [img], 'label': [label]})['image'][0]
        
#         processed_imgs.append(img_tensor)
#         labels.append(label)
    
#     # Create full dataset
#     full_dataset = TensorDataset(torch.stack(processed_imgs), torch.tensor(labels, dtype=torch.long))
    
#     # Partition data for federated learning
#     # Simple approach: divide data equally among clients
#     total_samples = len(full_dataset)
#     samples_per_client = total_samples // num_partitions
#     start_idx = partition_id * samples_per_client
    
#     if partition_id == num_partitions - 1:  # Last client gets remaining samples
#         end_idx = total_samples
#     else:
#         end_idx = start_idx + samples_per_client
    
#     # Create client's partition
#     client_indices = list(range(start_idx, end_idx))
#     client_dataset = Subset(full_dataset, client_indices)
    
#     # Split into train/test
#     client_size = len(client_dataset)
#     train_size = int(0.8 * client_size)
#     test_size = client_size - train_size
    
#     train_indices = client_indices[:train_size]
#     test_indices = client_indices[train_size:]
    
#     train_subset = Subset(full_dataset, train_indices)
#     test_subset = Subset(full_dataset, test_indices)
    
#     # Create DataLoaders
#     trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
#     testloader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
#     print(f"✅ Client {partition_id}: {len(train_subset)} train, {len(test_subset)} test samples")
    
#     return trainloader, testloader

# def train(net, trainloader, epochs, lr, device):
#     # Set matplotlib to use non-GUI backend to avoid threading issues
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
    
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     net.train()
#     train_losses = []

#     for epoch in range(epochs):
#         running_loss = 0.0
#         for batch in trainloader:
#             images, labels = batch["image"].to(device), batch["label"].to(device)
#             optimizer.zero_grad()
#             outputs = net(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         train_losses.append(running_loss / len(trainloader))

#     # Log training curve
#     try:
#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.plot(train_losses, label="Train Loss")
#         ax.set_title("Client Training Loss Curve")
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("Loss")
#         ax.legend()
#         mlflow.log_figure(fig, "client_train_loss.png")
#         plt.close(fig)
#     except Exception as e:
#         print(f"Training loss plotting failed: {str(e)}")

#     return train_losses[-1]

# def test(net, testloader, device, tag="client", round_num=None):
#     # Set matplotlib to use non-GUI backend to avoid threading issues
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
    
#     net.to(device)
#     net.eval()
#     criterion = nn.CrossEntropyLoss()
#     y_true, y_pred, scores = [], [], []
#     loss = 0.0

#     with torch.no_grad():
#         for batch in testloader:
#             # Handle both dictionary format (client) and tuple format (server)
#             if isinstance(batch, dict):
#                 # Dictionary format from HuggingFace datasets
#                 images = batch["image"].to(device)
#                 labels = batch["label"].to(device)
#             else:
#                 # Tuple format from TensorDataset
#                 images, labels = batch
#                 images = images.to(device)
#                 labels = labels.to(device)
            
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             preds = torch.argmax(outputs, dim=1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(preds.cpu().numpy())
#             scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())

#     # Metrics
#     accuracy = np.mean(np.array(y_pred) == np.array(y_true))
#     precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
#     recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
#     f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

#     # Log to MLflow with custom run name
#     custom_run_name = generate_run_name("PathMNIST_CNN", tag, round_num)
    
#     with mlflow.start_run(run_name=custom_run_name, nested=True):
#         mlflow.set_tag("model_type", tag)
#         mlflow.set_tag("model_name", "PathMNIST_CNN")
#         mlflow.set_tag("dataset", "PathMNIST")
#         if round_num is not None:
#             mlflow.set_tag("round", round_num)
            
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_metric("precision", precision)
#         mlflow.log_metric("recall", recall)
#         mlflow.log_metric("f1", f1)
#         mlflow.log_metric("loss", loss / len(testloader))

#         # Confusion Matrix
#         try:
#             cm = confusion_matrix(y_true, y_pred)
#             disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#             fig, ax = plt.subplots(figsize=(8, 6))
#             disp.plot(ax=ax)
#             plt.title(f"Confusion Matrix - {tag}")
#             mlflow.log_figure(fig, f"conf_matrix_{tag}.png")
#             plt.close(fig)
#         except Exception as e:
#             print(f"Confusion matrix plotting failed: {str(e)}")

#         # ROC Curve (macro-average)
#         try:
#             # PathMNIST has 9 classes
#             y_true_bin = np.eye(9)[y_true]  
#             fpr, tpr, _ = roc_curve(y_true_bin.ravel(), np.array(scores).ravel())
#             fig, ax = plt.subplots(figsize=(8, 6))
#             ax.plot(fpr, tpr, label="ROC curve")
#             ax.set_title(f"ROC Curve - {tag}")
#             ax.set_xlabel("False Positive Rate")
#             ax.set_ylabel("True Positive Rate")
#             ax.legend()
#             mlflow.log_figure(fig, f"roc_curve_{tag}.png")
#             plt.close(fig)
#         except Exception as e:
#             print(f"ROC curve plotting failed: {str(e)}")

#     return loss / len(testloader), accuracy

# def get_weights(net):
#     return [val.cpu().numpy() for _, val in net.state_dict().items()]


# def set_weights(net, parameters):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)
