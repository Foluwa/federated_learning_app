import random
import torch
import mlflow
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federated_learning_app.task import Net, get_weights, load_data, set_weights, test, train, generate_run_name
from federated_learning_app.knowledge_distillation import DEVICE_PROFILES


class FlowerClient(NumPyClient):
    """Enhanced client that adapts to different device types"""
    
    def __init__(self, net, trainloader, valloader, local_epochs, client_id, device_type='mobile'):
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.client_id = client_id
        self.device_type = device_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.current_round = 0
        
        # Create device-specific model
        self.net = Net(device_type)
        self.net.to(self.device)
        
        # Log device info
        profile = DEVICE_PROFILES.get(device_type, {})
        print(f"👤 Client {client_id}: {profile.get('name', device_type)} "
              f"({sum(p.numel() for p in self.net.parameters()):,} params)")

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        # Get round number and device-specific config
        round_num = config.get("round", self.current_round)
        self.current_round = round_num
        
        # Get device-specific training parameters
        device_local_epochs = config.get("local_epochs", self.local_epochs)
        device_lr = config.get("lr", 0.01)
        
        # Log device-specific training
        profile = DEVICE_PROFILES.get(self.device_type, {})
        print(f"🔧 Client {self.client_id} ({profile.get('name', self.device_type)}) "
              f"training for {device_local_epochs} epochs")
        
        train_loss = train(
            self.net,
            self.trainloader,
            device_local_epochs,
            device_lr,
            self.device,
            round_num=round_num
        )
        
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "device_type": self.device_type,
                "client_id": self.client_id
            },
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        
        # Get round number
        round_num = config.get("round", self.current_round)
        
        loss, accuracy = test(
            self.net, 
            self.valloader, 
            self.device, 
            tag=f"client_{self.client_id}_{self.device_type}",
            round_num=round_num
        )
        
        return loss, len(self.valloader.dataset), {
            "accuracy": accuracy,
            "device_type": self.device_type,
            "client_id": self.client_id
        }


def assign_device_type(client_id, num_clients, mode='dev'):
    """Assign device type to client based on realistic distribution"""
    
    # Device distributions by mode
    device_distributions = {
        'dev': {
            'devices': ['raspberry_pi', 'mobile'],
            'weights': [0.7, 0.3]  # 70% Pi, 30% mobile
        },
        'test': {
            'devices': ['raspberry_pi', 'mobile', 'workstation'],
            'weights': [0.5, 0.3, 0.2]  # 50% Pi, 30% mobile, 20% workstation
        },
        'prod': {
            'devices': ['raspberry_pi', 'mobile', 'workstation', 'server'],
            'weights': [0.4, 0.3, 0.2, 0.1]  # 40% Pi, 30% mobile, 20% workstation, 10% server
        }
    }
    
    distribution = device_distributions.get(mode, device_distributions['dev'])
    
    # Use client_id as seed for consistent assignment
    random.seed(client_id)
    device_type = random.choices(distribution['devices'], weights=distribution['weights'])[0]
    
    return device_type


def client_fn(context: Context):
    """Enhanced client function with device type assignment"""
    
    # Get client configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]
    
    # Assign device type based on client ID and mode
    mode = context.run_config.get("mode", "dev")
    device_type = assign_device_type(partition_id, num_partitions, mode)
    
    # Load data
    trainloader, valloader = load_data(partition_id, num_partitions)
    
    # Create device-specific client
    return FlowerClient(
        net=None, 
        trainloader=trainloader, 
        valloader=valloader, 
        local_epochs=local_epochs,
        client_id=partition_id,
        device_type=device_type
    ).to_client()

app = ClientApp(client_fn=client_fn)