from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from federated_learning_app.task import get_weights, set_weights, test, generate_run_name
from federated_learning_app.knowledge_distillation import DEVICE_PROFILES, TeacherModel, StudentModel, KnowledgeDistiller
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
        grayscale = np.dot(img_array, [0.299, 0.587, 0.114])
        return grayscale.astype(np.uint8)
    return img_array


from federated_learning_app.config_manager import ConfigManager
from federated_learning_app.knowledge_distillation import MultiDeviceKnowledgeDistiller, create_student_model
from federated_learning_app.device_manager import DeviceManager, DevicePerformanceTracker


def create_centralized_teacher_data(dataset_size=-1):
    """Create centralized dataset for teacher training with dynamic size"""
    print(f"📊 Loading centralized PathMNIST data for teacher training...")
    
    transform = Compose([
        ToTensor(), 
        Normalize(mean=[.5], std=[.5])
    ])
    
    # Load training data for teacher
    raw_dataset = PathMNIST(split="train", download=True)
    
    # Determine actual dataset size
    total_available = len(raw_dataset.imgs)
    if dataset_size == -1 or dataset_size >= total_available:
        actual_size = total_available
        print(f"   Using full dataset: {actual_size:,} samples")
    else:
        actual_size = min(dataset_size, total_available)
        print(f"   Using subset: {actual_size:,} / {total_available:,} samples")
    
    # Convert to grayscale and create tensors (with subset if needed)
    processed_imgs = []
    labels = []
    
    for i in range(actual_size):
        img_array = raw_dataset.imgs[i]
        label = raw_dataset.labels[i].item()
        
        gray_img = rgb_to_grayscale_numpy(img_array)
        img_tensor = transform(gray_img)
        processed_imgs.append(img_tensor)
        labels.append(label)
    
    imgs_tensor = torch.stack(processed_imgs)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    train_dataset = TensorDataset(imgs_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"✅ Teacher training data: {len(train_dataset):,} samples, {len(train_loader)} batches")
    return train_loader


def create_test_data(dataset_size=-1):
    """Create test dataset for evaluation with dynamic size"""
    transform = Compose([
        ToTensor(), 
        Normalize(mean=[.5], std=[.5])
    ])
    
    raw_dataset = PathMNIST(split="test", download=True)
    
    # Determine test dataset size (typically smaller subset for faster evaluation)
    total_available = len(raw_dataset.imgs)
    if dataset_size == -1:
        actual_size = total_available
    else:
        # Use proportionally smaller test set for dev/test modes
        test_ratio = 0.1  # 10% of training size for test
        actual_size = min(max(int(dataset_size * test_ratio), 100), total_available)
    
    processed_imgs = []
    labels = []
    
    for i in range(actual_size):
        img_array = raw_dataset.imgs[i]
        label = raw_dataset.labels[i].item()
        
        gray_img = rgb_to_grayscale_numpy(img_array)
        img_tensor = transform(gray_img)
        processed_imgs.append(img_tensor)
        labels.append(label)
    
    imgs_tensor = torch.stack(processed_imgs)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    test_dataset = TensorDataset(imgs_tensor, labels_tensor)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"📊 Test data: {len(test_dataset):,} samples for evaluation")
    return testloader


def get_evaluate_fn_multi_device(testloader, device, device_manager):
    """Enhanced evaluation function that can handle different device models"""
    def evaluate(server_round, parameters_ndarrays, config):
        # Use mobile model as default for server evaluation
        net = create_student_model('mobile', num_classes=9)
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device, tag="central", round_num=server_round)
        return loss, {"cen_accuracy": accuracy}
    return evaluate


def create_heterogeneous_strategy(device_manager, initial_parameters_mobile, fraction_fit, testloader, device):
    """Create federated learning strategy that handles heterogeneous devices"""
    
    class HeterogeneousFedAvg(FedAvg):
        """Extended FedAvg strategy for heterogeneous devices with performance tracking"""
        
        def __init__(self, device_manager, num_rounds=3, **kwargs):
            super().__init__(**kwargs)
            self.device_manager = device_manager
            self.performance_tracker = DevicePerformanceTracker()
            self.num_rounds = num_rounds
        
        def configure_fit(self, server_round, parameters, client_manager):
            """Configure fit with device-specific parameters"""
            config = super().configure_fit(server_round, parameters, client_manager)
            
            # Add device assignment info to config
            if hasattr(config, 'config'):
                config.config['server_round'] = server_round
                config.config['device_assignments'] = self.device_manager.device_assignments
            
            return config
        
        def aggregate_fit(self, server_round, results, failures):
            """Aggregate results from heterogeneous devices with performance tracking"""
            print(f"\n🔗 Aggregating Round {server_round} Results:")
            print(f"   - Received: {len(results)} results, {len(failures)} failures")
            
            # Track training metrics by device type
            for client_proxy, fit_res in results:
                # Extract client metrics from fit result
                metrics = fit_res.metrics
                
                # Get device info from metrics if available
                device_type = metrics.get('device_type', 'unknown')
                client_id = metrics.get('client_id', 'unknown')
                train_loss = metrics.get('train_loss', 0)
                
                if device_type != 'unknown':
                    # Record training metrics
                    training_metrics = {
                        'train_loss': train_loss,
                        'loss': train_loss,  # Use train_loss as proxy for loss
                        'accuracy': 0.5,  # Placeholder - would need actual training accuracy
                    }
                    
                    self.performance_tracker.record_client_metrics(
                        client_id, device_type, server_round, training_metrics
                    )
            
            # Use standard FedAvg aggregation
            return super().aggregate_fit(server_round, results, failures)
        
        def aggregate_evaluate(self, server_round, results, failures):
            """Aggregate evaluation results with device tracking"""
            print(f"📊 Round {server_round} Evaluation:")
            print(f"   - Evaluated: {len(results)} clients, {len(failures)} failures")
            
            # Track evaluation metrics by device type
            device_metrics = {}
            
            for client_proxy, evaluate_res in results:
                # Extract evaluation metrics
                metrics = evaluate_res.metrics
                
                device_type = metrics.get('device_type', 'unknown')
                client_id = metrics.get('client_id', 'unknown')
                accuracy = metrics.get('accuracy', 0)
                loss = evaluate_res.loss
                
                if device_type != 'unknown':
                    # Record evaluation metrics
                    eval_metrics = {
                        'accuracy': accuracy,
                        'loss': loss,
                        'train_loss': 0,  # Not available in evaluation
                    }
                    
                    self.performance_tracker.record_client_metrics(
                        client_id, device_type, server_round, eval_metrics
                    )
                    
                    # Group for immediate reporting
                    if device_type not in device_metrics:
                        device_metrics[device_type] = []
                    device_metrics[device_type].append({
                        'client_id': client_id,
                        'accuracy': accuracy,
                        'loss': loss
                    })
            
            # Print round summary by device type
            if device_metrics:
                print(f"   📱 Device Performance Summary:")
                for device_type, metrics in device_metrics.items():
                    avg_acc = np.mean([m['accuracy'] for m in metrics])
                    avg_loss = np.mean([m['loss'] for m in metrics])
                    profile = DEVICE_PROFILES.get(device_type, {})
                    device_name = profile.get('name', device_type)
                    
                    print(f"      {device_name:20}: {avg_acc:>6.1%} accuracy, "
                          f"{avg_loss:>6.3f} loss ({len(metrics)} clients)")
            
            # Generate performance report after final round
            if server_round == self.num_rounds:
                print(f"\n" + "="*80)
                print(f"🏁 FINAL PERFORMANCE ANALYSIS")
                print(f"="*80)
                self.performance_tracker.print_performance_report()
                
                # Save detailed analysis
                filename = self.performance_tracker.save_metrics_to_file(
                    f"device_analysis_round_{server_round}.json"
                )
                print(f"\n📊 Complete analysis saved: {filename}")
            
            return super().aggregate_evaluate(server_round, results, failures)
    
    return HeterogeneousFedAvg(
        device_manager=device_manager,
        num_rounds=3,  # Will be updated when called
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters_mobile,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda round_num: {
            "lr": 0.01 if round_num <= 2 else 0.005, 
            "round": round_num,
            "local_epochs": 1
        },
        evaluate_fn=get_evaluate_fn_multi_device(testloader, device, device_manager),
    )


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * metric["accuracy"] for num_examples, metric in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}


def on_fit_config(server_round: int) -> Metrics:
    lr = 0.01 if server_round <= 2 else 0.005
    return {"lr": lr, "round": server_round}


def server_fn(context: Context):
    """Enhanced server function with dynamic configuration support"""
    
    # Initialize dynamic configuration manager
    config_manager = ConfigManager(context)
    config = config_manager.config
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create parent MLflow run with mode information
    experiment_name = generate_run_name(f"PathMNIST_KD_FL_{config['mode'].upper()}", "federated_experiment")
    
    with mlflow.start_run(run_name=experiment_name):
        # Log all configuration parameters
        mlflow.set_tag("experiment_type", "knowledge_distillation_federated")
        mlflow.set_tag("model_name", "PathMNIST_Student_CNN")
        mlflow.set_tag("dataset", "PathMNIST")
        mlflow.set_tag("compression_enabled", True)
        mlflow.set_tag("runtime_mode", config['mode'])
        mlflow.set_tag("use_cached_models", config['use_cached_models'])
        mlflow.set_tag("force_retrain", config['force_retrain'])
        
        # Log all mode-specific parameters
        for key, value in config.items():
            if key not in ['use_cached_models', 'force_retrain', 'cache_dir']:
                mlflow.log_param(f"config_{key}", value)
        
        mlflow.log_param("device", str(device))
        
        print("🚀 Starting Knowledge Distillation + Federated Learning Pipeline")
        print("="*70)
        
        # Display comprehensive configuration
        config_manager.print_config()
        
        # ============================================================================
        # PHASE 1: MULTI-DEVICE KNOWLEDGE DISTILLATION
        # ============================================================================
        
        print("\n📚 Phase 1: Multi-Device Knowledge Distillation")
        print("-" * 50)
        
        # Initialize device manager
        device_manager = DeviceManager(config)
        target_devices = device_manager.get_target_devices()
        
        print(f"🎯 Target Device Types: {target_devices}")
        for device_type in target_devices:
            profile = DEVICE_PROFILES[device_type]
            print(f"   - {profile['name']}: {profile['description']}")
        
        # Assign devices to clients
        num_simulated_clients = 10  # From pyproject.toml num-supernodes
        device_assignments = device_manager.assign_devices_to_clients(num_simulated_clients)
        
        # Initialize multi-device knowledge distiller
        cache_suffix = config_manager.get_cache_suffix()
        distiller = MultiDeviceKnowledgeDistiller(
            device=device, 
            cache_dir=config['cache_dir'],
            cache_suffix=cache_suffix
        )
        
        # Determine training strategy
        should_load_from_cache = (config['use_cached_models'] and 
                                 not config['force_retrain'])
        
        if should_load_from_cache:
            print(f"\n⚡ Checking cached models for {config['mode']} mode...")
            
        # Train or load multi-device models
        if config['force_retrain'] or not should_load_from_cache:
            # Create teacher training data
            print("📊 Preparing training data...")
            teacher_train_loader = create_centralized_teacher_data(config['dataset_size'])
            
            # Train models for all target devices
            teacher_model, device_models = distiller.get_or_train_multi_device_models(
                train_loader=teacher_train_loader,
                config=config,
                target_devices=target_devices,
                use_cache=config['use_cached_models'],
                force_retrain=config['force_retrain']
            )
        else:
            print("⚡ Attempting to load cached models...")
            # Try to load cached models
            try:
                teacher_model, device_models = distiller.get_or_train_multi_device_models(
                    train_loader=None,
                    config=config,
                    target_devices=target_devices,
                    use_cache=True,
                    force_retrain=False
                )
            except:
                print("⚠️  Cache loading failed, training from scratch...")
                teacher_train_loader = create_centralized_teacher_data(config['dataset_size'])
                teacher_model, device_models = distiller.get_or_train_multi_device_models(
                    train_loader=teacher_train_loader,
                    config=config,
                    target_devices=target_devices,
                    use_cache=config['use_cached_models'],
                    force_retrain=True
                )
        
        # Set device models in device manager
        device_manager.set_device_models(device_models)
        
        # ============================================================================
        # MODEL ANALYSIS AND LOGGING
        # ============================================================================
        
        # Analyze all device models
        print(f"\n📊 Multi-Device Model Analysis ({config['mode'].upper()} mode):")
        teacher_params, teacher_size = sum(p.numel() for p in teacher_model.parameters()), sum(p.numel() * 4 for p in teacher_model.parameters()) / (1024 * 1024)
        
        print(f"🎓 Teacher Model: {teacher_params:,} params, {teacher_size:.2f} MB")
        print(f"🔧 Device Models:")
        
        total_compression_sum = 0
        for device_type, model in device_models.items():
            params = sum(p.numel() for p in model.parameters())
            size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
            compression = params / teacher_params
            total_compression_sum += compression
            
            profile = DEVICE_PROFILES[device_type]
            print(f"   {profile['name']:20}: {params:>8,} params ({compression:>6.1%} of teacher)")
            
            # Log to MLflow
            mlflow.log_param(f"{device_type}_params", params)
            mlflow.log_metric(f"{device_type}_compression_ratio", compression)
        
        avg_compression = total_compression_sum / len(device_models)
        mlflow.log_metric("avg_compression_ratio", avg_compression)
        mlflow.log_param("target_devices", target_devices)
        
        # ============================================================================
        # PHASE 2: HETEROGENEOUS FEDERATED LEARNING
        # ============================================================================
        
        print(f"\n🔗 Phase 2: Heterogeneous Federated Learning ({config['mode'].upper()} mode)")
        print("-" * 50)
        
        # Display federated learning configuration
        print(f"🎯 Federated Learning Configuration:")
        print(f"   - Device types: {len(device_models)} different architectures")
        print(f"   - Server rounds: {config['num_server_rounds']}")
        print(f"   - Clients per round: {config['fraction_fit']}")
        print(f"   - Local epochs: {config['local_epochs']}")
        print(f"   - Runtime mode: {config['mode'].upper()}")
        
        # Use mobile model as the "common" model for aggregation
        # In practice, you might use more sophisticated aggregation
        mobile_model = device_models.get('mobile', list(device_models.values())[0])
        ndarrays = get_weights(mobile_model)
        parameters = ndarrays_to_parameters(ndarrays)
        
        # Create test data for server-side evaluation
        print("📊 Preparing test data for evaluation...")
        testloader = create_test_data(config['dataset_size'])
        
        # Create heterogeneous federated learning strategy
        print("🏗️  Setting up heterogeneous federation strategy...")
        strategy = create_heterogeneous_strategy(
            device_manager=device_manager,
            initial_parameters_mobile=parameters,
            fraction_fit=config['fraction_fit'],
            testloader=testloader,
            device=device
        )
        
        # Set the actual number of rounds in the strategy
        strategy.num_rounds = config['num_server_rounds']

        # Configure server with dynamic rounds
        server_config = ServerConfig(num_rounds=config['num_server_rounds'])
        
        # Log final federated learning configuration
        mlflow.log_param("federated_strategy", "HeterogeneousFedAvg")
        mlflow.log_param("min_available_clients", 2)
        mlflow.log_param("initial_model_type", "heterogeneous_mobile")
        mlflow.log_param("num_device_types", len(device_models))
        
        # Summary
        print(f"\n🚀 Ready to start heterogeneous federated learning!")
        print(f"   ✅ Mode: {config['mode'].upper()}")
        print(f"   ✅ Multi-device distillation: {'Loaded from cache' if should_load_from_cache else 'Completed'}")
        print(f"   ✅ Device models: {len(device_models)} types created")
        print(f"   ✅ Client assignment: {num_simulated_clients} clients across {len(target_devices)} device types")
        print(f"   ✅ Federated setup: {config['num_server_rounds']} rounds with heterogeneous models")
        print(f"   ✅ Training config: {config['teacher_epochs']}+{config['student_epochs']} epochs per device")
        if config['dataset_size'] != -1:
            print(f"   ✅ Dataset: {config['dataset_size']:,} samples (subset)")
        else:
            print(f"   ✅ Dataset: Full PathMNIST dataset")
        
        return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)

# from typing import List, Tuple
# from flwr.common import Context, ndarrays_to_parameters, Metrics
# from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from flwr.server.strategy import FedAvg
# from federated_learning_app.task import Net, get_weights, set_weights, test, generate_run_name
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision.transforms import Compose, ToTensor, Normalize
# from medmnist import PathMNIST
# import torch
# import mlflow
# import numpy as np
# from datetime import datetime


# def rgb_to_grayscale_numpy(img_array):
#     """Convert RGB numpy array to grayscale using standard weights"""
#     if len(img_array.shape) == 3 and img_array.shape[2] == 3:
#         # Apply standard RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
#         grayscale = np.dot(img_array, [0.299, 0.587, 0.114])
#         return grayscale.astype(np.uint8)
#     return img_array  # Already grayscale


# def get_evaluate_fn(testloader, device):
#     def evaluate(server_round, parameters_ndarrays, config):
#         net = Net()
#         set_weights(net, parameters_ndarrays)
#         net.to(device)
#         loss, accuracy = test(net, testloader, device, tag="central", round_num=server_round)
#         return loss, {"cen_accuracy": accuracy}
#     return evaluate


# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     accuracies = [num_examples * metric["accuracy"] for num_examples, metric in metrics]
#     total_examples = sum(num_examples for num_examples, _ in metrics)
#     return {"accuracy": sum(accuracies) / total_examples}


# def on_fit_config(server_round: int) -> Metrics:
#     lr = 0.01 if server_round <= 2 else 0.005
#     return {"lr": lr, "round": server_round}


# def server_fn(context: Context):
#     num_rounds = context.run_config["num-server-rounds"]
#     fraction_fit = context.run_config["fraction-fit"]

#     # Create parent MLflow run with custom name
#     experiment_name = generate_run_name("PathMNIST_CNN", "federated_experiment")
    
#     # Set parent run
#     with mlflow.start_run(run_name=experiment_name):
#         mlflow.set_tag("experiment_type", "federated_learning")
#         mlflow.set_tag("model_name", "PathMNIST_CNN")
#         mlflow.set_tag("dataset", "PathMNIST")
#         mlflow.set_tag("num_rounds", num_rounds)
#         mlflow.set_tag("fraction_fit", fraction_fit)
#         mlflow.log_param("num_rounds", num_rounds)
#         mlflow.log_param("fraction_fit", fraction_fit)

#         ndarrays = get_weights(Net())
#         parameters = ndarrays_to_parameters(ndarrays)

#         # Simple transform without Grayscale (we'll handle it manually)
#         transform = Compose([
#             ToTensor(), 
#             Normalize(mean=[.5], std=[.5])
#         ])
        
#         raw_dataset = PathMNIST(split="test", download=True)
        
#         # Convert images to grayscale manually before applying transforms
#         processed_imgs = []
#         for img_array in raw_dataset.imgs:
#             # Convert to grayscale if needed
#             gray_img = rgb_to_grayscale_numpy(img_array)
#             # Apply transforms
#             img_tensor = transform(gray_img)
#             processed_imgs.append(img_tensor)
        
#         imgs_tensor = torch.stack(processed_imgs)
#         labels_tensor = torch.tensor(raw_dataset.labels.squeeze(), dtype=torch.long)
#         test_dataset = TensorDataset(imgs_tensor, labels_tensor)
#         testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#         strategy = FedAvg(
#             fraction_fit=fraction_fit,
#             fraction_evaluate=1.0,
#             min_available_clients=2,
#             initial_parameters=parameters,
#             evaluate_metrics_aggregation_fn=weighted_average,
#             on_fit_config_fn=on_fit_config,
#             evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
#         )

#         config = ServerConfig(num_rounds=num_rounds)
#         return ServerAppComponents(strategy=strategy, config=config)

# app = ServerApp(server_fn=server_fn)
