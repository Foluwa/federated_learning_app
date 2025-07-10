import torch
import mlflow
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federated_learning_app.task import Net, get_weights, load_data, set_weights, test, train, generate_run_name

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, client_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.client_id = client_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.current_round = 0

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        # Get round number from config
        round_num = config.get("round", self.current_round)
        self.current_round = round_num
        
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config["lr"],
            self.device,
            round_num=round_num
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        
        # Get round number from config
        round_num = config.get("round", self.current_round)
        
        loss, accuracy = test(
            self.net, 
            self.valloader, 
            self.device, 
            tag=f"client_{self.client_id}",
            round_num=round_num
        )
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    
    return FlowerClient(
        net, 
        trainloader, 
        valloader, 
        local_epochs,
        client_id=partition_id
    ).to_client()

app = ClientApp(client_fn=client_fn)
