from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from federated_learning_app.task import Net, get_weights, set_weights, test, generate_run_name
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from medmnist import PathMNIST
import torch
import mlflow
import numpy as np
from datetime import datetime


def rgb_to_grayscale_numpy(img_array):
    """Convert RGB numpy array to grayscale using standard weights"""
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Apply standard RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
        grayscale = np.dot(img_array, [0.299, 0.587, 0.114])
        return grayscale.astype(np.uint8)
    return img_array  # Already grayscale


def get_evaluate_fn(testloader, device):
    def evaluate(server_round, parameters_ndarrays, config):
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device, tag="central", round_num=server_round)
        return loss, {"cen_accuracy": accuracy}
    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * metric["accuracy"] for num_examples, metric in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}


def on_fit_config(server_round: int) -> Metrics:
    lr = 0.01 if server_round <= 2 else 0.005
    return {"lr": lr, "round": server_round}


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Create parent MLflow run with custom name
    experiment_name = generate_run_name("PathMNIST_CNN", "federated_experiment")
    
    # Set parent run
    with mlflow.start_run(run_name=experiment_name):
        mlflow.set_tag("experiment_type", "federated_learning")
        mlflow.set_tag("model_name", "PathMNIST_CNN")
        mlflow.set_tag("dataset", "PathMNIST")
        mlflow.set_tag("num_rounds", num_rounds)
        mlflow.set_tag("fraction_fit", fraction_fit)
        mlflow.log_param("num_rounds", num_rounds)
        mlflow.log_param("fraction_fit", fraction_fit)

        ndarrays = get_weights(Net())
        parameters = ndarrays_to_parameters(ndarrays)

        # Simple transform without Grayscale (we'll handle it manually)
        transform = Compose([
            ToTensor(), 
            Normalize(mean=[.5], std=[.5])
        ])
        
        raw_dataset = PathMNIST(split="test", download=True)
        
        # Convert images to grayscale manually before applying transforms
        processed_imgs = []
        for img_array in raw_dataset.imgs:
            # Convert to grayscale if needed
            gray_img = rgb_to_grayscale_numpy(img_array)
            # Apply transforms
            img_tensor = transform(gray_img)
            processed_imgs.append(img_tensor)
        
        imgs_tensor = torch.stack(processed_imgs)
        labels_tensor = torch.tensor(raw_dataset.labels.squeeze(), dtype=torch.long)
        test_dataset = TensorDataset(imgs_tensor, labels_tensor)
        testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=on_fit_config,
            evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
        )

        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
