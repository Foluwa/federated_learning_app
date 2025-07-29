import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
try:
    import pandas as pd
except ImportError:
    print("⚠️  pandas not available, CSV export will be limited")
    pd = None

from federated_learning_app.knowledge_distillation import DEVICE_PROFILES

class DevicePerformanceAnalyzer:
    """Analyze and visualize device performance across tiers"""
    
    def __init__(self, analysis_file=None):
        self.analysis_data = None
        if analysis_file and Path(analysis_file).exists():
            self.load_analysis_file(analysis_file)
    
    def load_analysis_file(self, filename):
        """Load performance analysis from JSON file"""
        with open(filename, 'r') as f:
            self.analysis_data = json.load(f)
        print(f"📊 Loaded analysis data from: {filename}")
        return self.analysis_data
    
    def find_latest_analysis_file(self):
        """Find the most recent analysis file"""
        analysis_files = list(Path('.').glob('device_analysis_round_*.json'))
        if analysis_files:
            latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            self.load_analysis_file(latest_file)
            return latest_file
        return None
    
    def create_performance_comparison_chart(self, save_path="device_performance_comparison.png"):
        """Create visual comparison of device performance"""
        if not self.analysis_data:
            print("❌ No analysis data loaded")
            return
        
        summary = self.analysis_data['summary']
        
        # Prepare data for plotting
        devices = []
        accuracies = []
        std_devs = []
        target_times = []
        memory_limits = []
        param_counts = []
        
        for device_type, stats in summary.items():
            profile = stats['device_profile']
            latest = stats['latest_round']
            
            devices.append(profile['name'])
            accuracies.append(latest['avg_accuracy'])
            std_devs.append(latest['std_accuracy'])
            target_times.append(profile['target_inference_ms'])
            memory_limits.append(profile['memory_mb'])
            
            # Calculate approximate parameter count (would need actual model info)
            # This is an estimate based on device profile
            if 'raspberry_pi' in device_type:
                param_counts.append(2345)
            elif 'mobile' in device_type:
                param_counts.append(10901)
            elif 'workstation' in device_type:
                param_counts.append(30000)
            else:  # server
                param_counts.append(50000)
        
        # Create comprehensive comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Device Tier Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison with error bars
        ax1.bar(range(len(devices)), accuracies, yerr=std_devs, capsize=5, 
                color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(devices)])
        ax1.set_xlabel('Device Type')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by Device Type')
        ax1.set_xticks(range(len(devices)))
        ax1.set_xticklabels(devices, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add accuracy values on bars
        for i, (acc, std) in enumerate(zip(accuracies, std_devs)):
            ax1.text(i, acc + std + 0.01, f'{acc:.1%}', ha='center', va='bottom')
        
        # 2. Resource efficiency (accuracy per inference time constraint)
        efficiency = [acc / (time/1000) for acc, time in zip(accuracies, target_times)]
        ax2.bar(range(len(devices)), efficiency, 
                color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f'][:len(devices)])
        ax2.set_xlabel('Device Type')
        ax2.set_ylabel('Accuracy per Second')
        ax2.set_title('Resource Efficiency (Accuracy/Target Time)')
        ax2.set_xticks(range(len(devices)))
        ax2.set_xticklabels(devices, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory vs Performance scatter
        scatter = ax3.scatter(memory_limits, accuracies, s=[p/100 for p in param_counts], 
                            c=target_times, cmap='RdYlBu_r', alpha=0.7)
        ax3.set_xlabel('Memory Limit (MB)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Memory Constraint vs Performance')
        ax3.grid(True, alpha=0.3)
        
        # Add device labels
        for i, device in enumerate(devices):
            ax3.annotate(device, (memory_limits[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar for inference time
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Target Inference Time (ms)')
        
        # 4. Performance vs Model Size
        ax4.scatter(param_counts, accuracies, s=100, 
                   c=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(devices)])
        ax4.set_xlabel('Model Parameters')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Model Size vs Performance')
        ax4.grid(True, alpha=0.3)
        
        # Add device labels
        for i, device in enumerate(devices):
            ax4.annotate(device, (param_counts[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance comparison chart saved: {save_path}")
        plt.show()
        
        return fig
    
    def create_round_progression_chart(self, save_path="device_round_progression.png"):
        """Create chart showing performance progression across rounds"""
        if not self.analysis_data:
            print("❌ No analysis data loaded")
            return
        
        round_data = self.analysis_data['round_comparison']
        
        # Prepare data
        rounds = sorted([int(r) for r in round_data.keys()])
        device_types = set()
        for round_metrics in round_data.values():
            device_types.update(round_metrics.keys())
        device_types = sorted(list(device_types))
        
        # Create progression plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Performance Progression Across Federated Learning Rounds', fontsize=14)
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        
        # Accuracy progression
        for i, device_type in enumerate(device_types):
            device_accuracies = []
            device_rounds = []
            
            for round_num in rounds:
                if str(round_num) in round_data and device_type in round_data[str(round_num)]:
                    device_accuracies.append(round_data[str(round_num)][device_type]['avg_accuracy'])
                    device_rounds.append(round_num)
            
            if device_accuracies:
                profile = DEVICE_PROFILES.get(device_type, {'name': device_type})
                ax1.plot(device_rounds, device_accuracies, 'o-', 
                        color=colors[i % len(colors)], label=profile['name'], linewidth=2)
        
        ax1.set_xlabel('Federated Learning Round')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Progression by Device Type')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss progression
        for i, device_type in enumerate(device_types):
            device_losses = []
            device_rounds = []
            
            for round_num in rounds:
                if str(round_num) in round_data and device_type in round_data[str(round_num)]:
                    device_losses.append(round_data[str(round_num)][device_type]['avg_loss'])
                    device_rounds.append(round_num)
            
            if device_losses:
                profile = DEVICE_PROFILES.get(device_type, {'name': device_type})
                ax2.plot(device_rounds, device_losses, 'o-', 
                        color=colors[i % len(colors)], label=profile['name'], linewidth=2)
        
        ax2.set_xlabel('Federated Learning Round')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Progression by Device Type')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Round progression chart saved: {save_path}")
        plt.show()
        
        return fig
    
    def generate_performance_report(self):
        """Generate comprehensive text report"""
        if not self.analysis_data:
            print("❌ No analysis data loaded")
            return
        
        summary = self.analysis_data['summary']
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE DEVICE PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Overall statistics
        total_clients = sum(stats['unique_clients'] for stats in summary.values())
        avg_accuracy = np.mean([stats['latest_round']['avg_accuracy'] for stats in summary.values()])
        
        print(f"📋 Experiment Overview:")
        print(f"   - Total device types: {len(summary)}")
        print(f"   - Total unique clients: {total_clients}")
        print(f"   - Average accuracy across all devices: {avg_accuracy:.1%}")
        print()
        
        # Device-specific analysis
        print(f"🔧 Device Type Analysis:")
        print("-" * 80)
        
        # Sort by target inference time (most constrained first)
        sorted_devices = sorted(summary.items(), 
                              key=lambda x: x[1]['device_profile']['target_inference_ms'], 
                              reverse=True)
        
        for device_type, stats in sorted_devices:
            profile = stats['device_profile']
            latest = stats['latest_round']
            
            print(f"\n📱 {profile['name']} ({device_type})")
            print(f"   Hardware Constraints:")
            print(f"     - Memory limit: {profile['memory_mb']:,} MB")
            print(f"     - Target inference: {profile['target_inference_ms']} ms")
            print(f"     - Priority: {profile['priority']}")
            print(f"   Performance Results:")
            print(f"     - Clients: {stats['unique_clients']}")
            print(f"     - Average accuracy: {latest['avg_accuracy']:.1%}")
            print(f"     - Accuracy range: {latest['min_accuracy']:.1%} - {latest['max_accuracy']:.1%}")
            print(f"     - Standard deviation: {latest['std_accuracy']:.1%}")
            print(f"     - Average loss: {latest['avg_loss']:.3f}")
            
            # Performance assessment
            if latest['avg_accuracy'] >= 0.4:  # 40% for 9-class problem
                assessment = "🟢 EXCELLENT"
            elif latest['avg_accuracy'] >= 0.3:  # 30%
                assessment = "🟡 GOOD"
            elif latest['avg_accuracy'] >= 0.2:  # 20%
                assessment = "🟠 ACCEPTABLE"
            else:
                assessment = "🔴 NEEDS IMPROVEMENT"
            
            print(f"     - Assessment: {assessment}")
        
        # Comparative analysis
        print(f"\n🏆 Comparative Analysis:")
        print("-" * 50)
        
        # Best performer
        best_device = max(sorted_devices, key=lambda x: x[1]['latest_round']['avg_accuracy'])
        best_acc = best_device[1]['latest_round']['avg_accuracy']
        print(f"Best performer: {best_device[1]['device_profile']['name']} ({best_acc:.1%})")
        
        # Most efficient (accuracy per resource constraint)
        efficiency_scores = []
        for device_type, stats in sorted_devices:
            profile = stats['device_profile']
            acc = stats['latest_round']['avg_accuracy']
            # Simple efficiency: accuracy / (memory_gb * inference_time_sec)
            efficiency = acc / ((profile['memory_mb']/1000) * (profile['target_inference_ms']/1000))
            efficiency_scores.append((device_type, efficiency, profile['name']))
        
        most_efficient = max(efficiency_scores, key=lambda x: x[1])
        print(f"Most efficient: {most_efficient[2]} (efficiency score: {most_efficient[1]:.2f})")
        
        # Performance gap analysis
        accuracies = [stats['latest_round']['avg_accuracy'] for _, stats in sorted_devices]
        performance_gap = max(accuracies) - min(accuracies)
        print(f"Performance gap: {performance_gap:.1%}")
        
        if performance_gap > 0.15:  # 15%
            print("   ⚠️  Significant performance gap - consider model architecture adjustments")
        elif performance_gap > 0.1:  # 10%
            print("   📊 Moderate performance gap - expected for heterogeneous deployment")
        else:
            print("   ✅ Low performance gap - well-balanced device optimization")
        
        print("\n" + "="*80)
    
    def compare_with_baseline(self, baseline_accuracy=0.16):
        """Compare device performance with baseline (e.g., random guessing)"""
        if not self.analysis_data:
            print("❌ No analysis data loaded")
            return
        
        summary = self.analysis_data['summary']
        
        print(f"\n📊 Improvement Over Baseline ({baseline_accuracy:.1%}):")
        print("-" * 50)
        
        for device_type, stats in summary.items():
            profile = stats['device_profile']
            accuracy = stats['latest_round']['avg_accuracy']
            improvement = accuracy - baseline_accuracy
            improvement_factor = accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
            
            print(f"{profile['name']:20}: {accuracy:.1%} ({improvement:+.1%}, {improvement_factor:.1f}x baseline)")
    
    def export_to_csv(self, filename="device_performance_data.csv"):
        """Export performance data to CSV for further analysis"""
        if not self.analysis_data:
            print("❌ No analysis data loaded")
            return
        
        # Prepare data for CSV
        rows = []
        for device_type, metrics_list in self.analysis_data['raw_metrics'].items():
            profile = DEVICE_PROFILES.get(device_type, {})
            
            for record in metrics_list:
                rows.append({
                    'device_type': device_type,
                    'device_name': profile.get('name', device_type),
                    'client_id': record['client_id'],
                    'round': record['round'],
                    'accuracy': record['accuracy'],
                    'loss': record['loss'],
                    'train_loss': record['train_loss'],
                    'memory_limit_mb': profile.get('memory_mb', 0),
                    'target_inference_ms': profile.get('target_inference_ms', 0),
                    'priority': profile.get('priority', 'unknown'),
                    'timestamp': record['timestamp']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"📄 Performance data exported to CSV: {filename}")
        return df


def analyze_latest_experiment():
    """Quick function to analyze the most recent experiment"""
    analyzer = DevicePerformanceAnalyzer()
    latest_file = analyzer.find_latest_analysis_file()
    
    if latest_file:
        print(f"🔍 Analyzing latest experiment: {latest_file}")
        analyzer.generate_performance_report()
        analyzer.compare_with_baseline(0.16)  # Random baseline for 9-class problem
        analyzer.create_performance_comparison_chart()
        analyzer.create_round_progression_chart()
        analyzer.export_to_csv()
        return analyzer
    else:
        print("❌ No analysis files found. Run a federated learning experiment first.")
        return None


def quick_performance_check():
    """Quick performance check without plots"""
    analyzer = DevicePerformanceAnalyzer()
    latest_file = analyzer.find_latest_analysis_file()
    
    if latest_file:
        analyzer.generate_performance_report()
        return analyzer
    else:
        print("❌ No analysis files found.")
        return None


if __name__ == "__main__":
    # Run analysis if script is executed directly
    analyzer = analyze_latest_experiment()