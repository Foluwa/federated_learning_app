import random
import numpy as np
from datetime import datetime
from federated_learning_app.knowledge_distillation import DEVICE_PROFILES

class DeviceManager:
    """Manages device assignment and model distribution"""
    
    def __init__(self, config):
        self.config = config
        self.device_assignments = {}
        self.device_models = {}
        
    def get_target_devices(self):
        """Get target device types based on configuration"""
        mode = self.config.get('mode', 'dev')
        
        if mode == 'dev':
            # Development: test 2 device types
            return ['raspberry_pi', 'mobile']
        elif mode == 'test':
            # Testing: test 3 device types
            return ['raspberry_pi', 'mobile', 'workstation']
        else:  # prod
            # Production: all device types
            return ['raspberry_pi', 'mobile', 'workstation', 'server']
    
    def assign_devices_to_clients(self, num_clients):
        """Assign device types to clients based on realistic distribution"""
        target_devices = self.get_target_devices()
        
        # Realistic distribution weights (more edge devices in real deployments)
        if len(target_devices) == 2:  # dev mode
            weights = [0.7, 0.3]  # 70% Pi, 30% mobile
        elif len(target_devices) == 3:  # test mode
            weights = [0.5, 0.3, 0.2]  # 50% Pi, 30% mobile, 20% workstation
        else:  # prod mode
            weights = [0.4, 0.3, 0.2, 0.1]  # 40% Pi, 30% mobile, 20% workstation, 10% server
        
        # Assign devices to clients
        assignments = {}
        for client_id in range(num_clients):
            device_type = np.random.choice(target_devices, p=weights)
            assignments[client_id] = device_type
        
        self.device_assignments = assignments
        
        # Print assignment summary
        self._print_device_assignments(assignments, target_devices)
        
        return assignments
    
    def _print_device_assignments(self, assignments, target_devices):
        """Print device assignment summary"""
        print(f"\n🔧 Device Assignment Summary")
        print("-" * 50)
        
        # Count assignments per device
        device_counts = {device: 0 for device in target_devices}
        for device_type in assignments.values():
            device_counts[device_type] += 1
        
        total_clients = len(assignments)
        for device_type, count in device_counts.items():
            profile = DEVICE_PROFILES[device_type]
            percentage = (count / total_clients) * 100
            print(f"{profile['name']:20} | {count:>2} clients ({percentage:>4.1f}%) | "
                  f"{profile['target_inference_ms']:>3}ms | {profile['memory_mb']:>4}MB")
        
        print(f"{'Total':20} | {total_clients:>2} clients")
        print("-" * 50)
    
    def get_client_device_type(self, client_id):
        """Get device type for specific client"""
        return self.device_assignments.get(client_id, 'mobile')  # Default to mobile
    
    def get_device_model(self, device_type):
        """Get model for specific device type"""
        return self.device_models.get(device_type)
    
    def set_device_models(self, device_models):
        """Set the trained device models"""
        self.device_models = device_models
        
        print(f"\n📱 Device Models Available:")
        for device_type, model in device_models.items():
            profile = DEVICE_PROFILES[device_type]
            params = sum(p.numel() for p in model.parameters())
            print(f"   {profile['name']:20} | {params:>8,} parameters")
    
    def get_device_config(self, device_type):
        """Get configuration for specific device type"""
        if device_type not in DEVICE_PROFILES:
            device_type = 'mobile'  # Fallback
        
        profile = DEVICE_PROFILES[device_type]
        return {
            'device_type': device_type,
            'batch_size': profile['batch_size'],
            'memory_mb': profile['memory_mb'],
            'target_inference_ms': profile['target_inference_ms'],
            'priority': profile['priority']
        }
    
    def get_federated_config_for_client(self, client_id, base_config):
        """Get federated learning config customized for client's device"""
        device_type = self.get_client_device_type(client_id)
        device_config = self.get_device_config(device_type)
        
        # Customize training parameters based on device capabilities
        custom_config = base_config.copy()
        
        # Adjust local epochs based on device capability
        if device_type == 'raspberry_pi':
            custom_config['local_epochs'] = max(1, base_config.get('local_epochs', 1))  # Minimum for efficiency
        elif device_type == 'server':
            custom_config['local_epochs'] = base_config.get('local_epochs', 1) * 2  # More epochs for powerful devices
        
        # Add device info
        custom_config.update({
            'device_type': device_type,
            'device_batch_size': device_config['batch_size'],
            'device_priority': device_config['priority']
        })
        
        return custom_config


class DevicePerformanceTracker:
    """Track and analyze performance across different device types"""
    
    def __init__(self):
        self.device_metrics = {}
        self.round_metrics = {}
        
    def record_client_metrics(self, client_id, device_type, round_num, metrics):
        """Record performance metrics for a client"""
        if device_type not in self.device_metrics:
            self.device_metrics[device_type] = []
        
        # Record client-specific metrics
        client_record = {
            'client_id': client_id,
            'round': round_num,
            'accuracy': metrics.get('accuracy', 0),
            'loss': metrics.get('loss', 0),
            'train_loss': metrics.get('train_loss', 0),
            'training_time': metrics.get('training_time', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        self.device_metrics[device_type].append(client_record)
        
        # Track round-wise aggregated metrics
        if round_num not in self.round_metrics:
            self.round_metrics[round_num] = {}
        
        if device_type not in self.round_metrics[round_num]:
            self.round_metrics[round_num][device_type] = []
            
        self.round_metrics[round_num][device_type].append(client_record)
    
    def get_device_performance_summary(self):
        """Get comprehensive performance summary by device type"""
        summary = {}
        
        for device_type, metrics_list in self.device_metrics.items():
            if not metrics_list:
                continue
                
            # Extract metrics
            accuracies = [m['accuracy'] for m in metrics_list]
            losses = [m['loss'] for m in metrics_list]
            train_losses = [m['train_loss'] for m in metrics_list]
            
            # Get latest round metrics for each client
            latest_metrics = {}
            for record in metrics_list:
                client_id = record['client_id']
                if client_id not in latest_metrics or record['round'] > latest_metrics[client_id]['round']:
                    latest_metrics[client_id] = record
            
            latest_accuracies = [m['accuracy'] for m in latest_metrics.values()]
            latest_losses = [m['loss'] for m in latest_metrics.values()]
            
            summary[device_type] = {
                'device_profile': DEVICE_PROFILES[device_type],
                'total_records': len(metrics_list),
                'unique_clients': len(latest_metrics),
                'all_rounds': {
                    'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                    'std_accuracy': np.std(accuracies) if accuracies else 0,
                    'avg_loss': np.mean(losses) if losses else 0,
                    'avg_train_loss': np.mean(train_losses) if train_losses else 0,
                },
                'latest_round': {
                    'avg_accuracy': np.mean(latest_accuracies) if latest_accuracies else 0,
                    'std_accuracy': np.std(latest_accuracies) if latest_accuracies else 0,
                    'avg_loss': np.mean(latest_losses) if latest_losses else 0,
                    'min_accuracy': min(latest_accuracies) if latest_accuracies else 0,
                    'max_accuracy': max(latest_accuracies) if latest_accuracies else 0,
                }
            }
        
        return summary
    
    def get_round_comparison(self):
        """Compare performance across rounds by device type"""
        round_comparison = {}
        
        for round_num, round_data in self.round_metrics.items():
            round_comparison[round_num] = {}
            
            for device_type, records in round_data.items():
                accuracies = [r['accuracy'] for r in records]
                losses = [r['loss'] for r in records]
                
                round_comparison[round_num][device_type] = {
                    'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                    'avg_loss': np.mean(losses) if losses else 0,
                    'client_count': len(records)
                }
        
        return round_comparison
    
    def print_performance_report(self):
        """Print comprehensive performance comparison across device types"""
        summary = self.get_device_performance_summary()
        
        print(f"\n📊 Device Performance Analysis Report")
        print("=" * 100)
        
        # Header
        print(f"{'Device Type':20} | {'Clients':>7} | {'Latest Acc':>10} | {'Acc Range':>15} | "
              f"{'Avg Loss':>8} | {'Target':>6} | {'Status':>8}")
        print("-" * 100)
        
        # Sort devices by target inference time (most constrained first)
        sorted_devices = sorted(summary.items(), 
                              key=lambda x: x[1]['device_profile']['target_inference_ms'], 
                              reverse=True)
        
        best_accuracy = 0
        for device_type, stats in sorted_devices:
            profile = stats['device_profile']
            latest = stats['latest_round']
            
            best_accuracy = max(best_accuracy, latest['avg_accuracy'])
            
            # Determine status
            target_acc = 0.3  # Minimum acceptable accuracy
            if latest['avg_accuracy'] >= target_acc:
                status = "✅ GOOD"
            else:
                status = "⚠️  LOW"
            
            print(f"{profile['name']:20} | "
                  f"{stats['unique_clients']:>7} | "
                  f"{latest['avg_accuracy']:>9.1%} | "
                  f"{latest['min_accuracy']:.1%}-{latest['max_accuracy']:.1%}      | "
                  f"{latest['avg_loss']:>8.3f} | "
                  f"{profile['target_inference_ms']:>4}ms | "
                  f"{status:>8}")
        
        print("-" * 100)
        print(f"{'BEST PERFORMANCE':20} | {'':>7} | {best_accuracy:>9.1%} | {'':>15} | {'':>8} | {'':>6} | {'':>8}")
        print("=" * 100)
        
        # Performance insights
        self._print_performance_insights(summary)
    
    def _print_performance_insights(self, summary):
        """Print insights about device performance"""
        print(f"\n🔍 Performance Insights:")
        
        # Find best and worst performing devices
        devices_by_accuracy = sorted(summary.items(), 
                                   key=lambda x: x[1]['latest_round']['avg_accuracy'], 
                                   reverse=True)
        
        if len(devices_by_accuracy) >= 2:
            best_device, best_stats = devices_by_accuracy[0]
            worst_device, worst_stats = devices_by_accuracy[-1]
            
            best_profile = best_stats['device_profile']
            worst_profile = worst_stats['device_profile']
            
            accuracy_gap = best_stats['latest_round']['avg_accuracy'] - worst_stats['latest_round']['avg_accuracy']
            
            print(f"   🥇 Best: {best_profile['name']} ({best_stats['latest_round']['avg_accuracy']:.1%} accuracy)")
            print(f"   📉 Worst: {worst_profile['name']} ({worst_stats['latest_round']['avg_accuracy']:.1%} accuracy)")
            print(f"   📊 Performance Gap: {accuracy_gap:.1%}")
            
            # Analysis
            if accuracy_gap > 0.1:  # 10% gap
                print(f"   ⚠️  Significant performance gap detected!")
                print(f"      Consider: More training epochs for {worst_profile['name']}")
            else:
                print(f"   ✅ Performance gap is reasonable ({accuracy_gap:.1%})")
        
        # Check if edge devices (Pi) are performing adequately
        if 'raspberry_pi' in summary:
            pi_accuracy = summary['raspberry_pi']['latest_round']['avg_accuracy']
            if pi_accuracy < 0.25:  # 25% threshold for 9-class problem
                print(f"   🚨 Raspberry Pi performance is very low ({pi_accuracy:.1%})")
                print(f"      Recommendation: Increase model capacity or training time")
            elif pi_accuracy >= 0.35:
                print(f"   🎉 Raspberry Pi achieving good performance ({pi_accuracy:.1%}) despite constraints!")
        
        # Memory efficiency analysis
        print(f"\n💾 Resource Efficiency:")
        for device_type, stats in summary.items():
            profile = stats['device_profile'] 
            accuracy = stats['latest_round']['avg_accuracy']
            # Simple efficiency metric: accuracy per MB of memory allowance
            efficiency = accuracy / (profile['memory_mb'] / 1000)  # per GB
            print(f"   {profile['name']:20}: {efficiency:.3f} accuracy/GB")

    def save_metrics_to_file(self, filename="device_performance_analysis.json"):
        """Save all metrics to JSON file for further analysis"""
        import json
        
        analysis_data = {
            'summary': self.get_device_performance_summary(),
            'round_comparison': self.get_round_comparison(),
            'raw_metrics': self.device_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        analysis_data = convert_numpy(analysis_data)
        
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"📄 Performance analysis saved to: {filename}")
        return filename