from collections import OrderedDict
from torchvision.transforms import Grayscale
from datetime import datetime
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Pad

import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Replace dataset in load_data
from medmnist import PathMNIST
from torchvision.transforms import Compose, ToTensor, Normalize

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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
    from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale
    
    transform = Compose([
        Grayscale(num_output_channels=1),  # Add this line
        ToTensor(), 
        Normalize((0.5,), (0.5,))
    ])
    
    def apply_transforms(batch):
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch
    return apply_transforms

def load_data(partition_id: int, num_partitions: int):
    global fds
    from flwr_datasets.partitioner import DirichletPartitioner
    from flwr_datasets import FederatedDataset

    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=1.0)
        fds = FederatedDataset(dataset="medmnist/pathmnist", partitioners={"train": partitioner})

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(get_transforms())
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader

def train(net, trainloader, epochs, lr, device):
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
            images, labels = batch["image"].to(device), batch["label"].to(device)
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
        ax.set_title("Client Training Loss Curve")
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
    custom_run_name = generate_run_name("PathMNIST_CNN", tag, round_num)
    
    with mlflow.start_run(run_name=custom_run_name, nested=True):
        mlflow.set_tag("model_type", tag)
        mlflow.set_tag("model_name", "PathMNIST_CNN")
        mlflow.set_tag("dataset", "PathMNIST")
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
            plt.title(f"Confusion Matrix - {tag}")
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
            ax.set_title(f"ROC Curve - {tag}")
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
