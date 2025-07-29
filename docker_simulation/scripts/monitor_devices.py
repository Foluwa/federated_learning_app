#!/usr/bin/env python3
"""
Real-time Device Monitoring for Federated Learning Simulation
"""

import subprocess
import time
import json
import os
import sys
from datetime import datetime
import argparse

class DeviceMonitor:
    """Monitor Docker-based device simulation in real-time"""
    
    def __init__(self):
        self.device_types = {
            'raspberry-pi': {'name': 'Raspberry Pi 4', 'emoji': '🔧', 'color': '\033[91m'},
            'mobile': {'name': 'Mobile Device', 'emoji': '📱', 'color': '\033[92m'},
            'workstation': {'name': 'Workstation', 'emoji': '🖥️', 'color': '\033[94m'},
            'server-device': {'name': 'Server Device', 'emoji': '🏥', 'color': '\033[95m'},
            'fl-server': {'name': 'FL Server', 'emoji': '🌐', 'color': '\033[96m'}
        }
        self.reset_color = '\033[0m'
    
    def get_container_stats(self):
        """Get real-time container statistics"""
        try:
            result = subprocess.run([
                'docker', 'stats', '--no-stream', '--format',
                'json'
            ], capture_output=True, text=True, check=True)
            
            stats = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        stats.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            return stats
        except subprocess.CalledProcessError:
            return []
    
    def get_network_stats(self):
        """Get network statistics for containers"""
        try:
            result = subprocess.run([
                'docker', 'network', 'inspect', 'docker_simulation_fl-network'
            ], capture_output=True, text=True, check=True)
            
            network_info = json.loads(result.stdout)[0]
            containers = network_info.get('Containers', {})
            
            return len(containers)
        except:
            return 0
    
    def parse_memory(self, mem_string):
        """Parse memory string like '512MiB / 2GiB' and return usage percentage"""
        try:
            used, total = mem_string.split(' / ')
            
            def to_bytes(s):
                s = s.strip()
                if s.endswith('KiB'):
                    return float(s[:-3]) * 1024
                elif s.endswith('MiB'):
                    return float(s[:-3]) * 1024 * 1024
                elif s.endswith('GiB'):
                    return float(s[:-3]) * 1024 * 1024 * 1024
                elif s.endswith('B'):
                    return float(s[:-1])
                return float(s)
            
            used_bytes = to_bytes(used)
            total_bytes = to_bytes(total)
            
            return (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0
        except:
            return 0
    
    def classify_container(self, container_name):
        """Classify container by device type"""
        for device_type, info in self.device_types.items():
            if device_type in container_name:
                return device_type
        return 'unknown'
    
    def display_dashboard(self):
        """Display real-time monitoring dashboard"""
        while True:
            try:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Get statistics
                stats = self.get_container_stats()
                network_count = self.get_network_stats()
                
                # Header
                print("🚀 Federated Learning Device Simulation Monitor")
                print("=" * 80)
                print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"🌐 Network containers: {network_count}")
                print(f"🔄 Refresh rate: 5 seconds (Ctrl+C to exit)")
                print()
                
                if not stats:
                    print("❌ No containers running or Docker not accessible")
                    time.sleep(5)
                    continue
                
                # Group by device type
                device_stats = {}
                for stat in stats:
                    container_name = stat['Name']
                    device_type = self.classify_container(container_name)
                    
                    if device_type not in device_stats:
                        device_stats[device_type] = []
                    
                    device_stats[device_type].append({
                        'name': container_name,
                        'cpu': stat['CPUPerc'].rstrip('%'),
                        'memory': stat['MemUsage'],
                        'memory_pct': self.parse_memory(stat['MemUsage']),
                        'network': stat['NetIO'],
                        'block': stat['BlockIO']
                    })
                
                # Display by device type
                for device_type, containers in device_stats.items():
                    if device_type == 'unknown':
                        continue
                        
                    device_info = self.device_types.get(device_type, {
                        'name': device_type, 'emoji': '❓', 'color': '\033[97m'
                    })
                    
                    print(f"{device_info['color']}{device_info['emoji']} {device_info['name']} "
                          f"({len(containers)} containers){self.reset_color}")
                    print("-" * 60)
                    
                    for container in containers:
                        # Color code by resource usage
                        cpu_val = float(container['cpu']) if container['cpu'] else 0
                        mem_val = container['memory_pct']
                        
                        cpu_color = self._get_usage_color(cpu_val)
                        mem_color = self._get_usage_color(mem_val)
                        
                        print(f"  📦 {container['name'][:25]:<25}")
                        print(f"     CPU: {cpu_color}{cpu_val:>6.1f}%{self.reset_color} | "
                              f"Memory: {mem_color}{mem_val:>6.1f}%{self.reset_color}")
                        print(f"     Network: {container['network']} | "
                              f"Disk: {container['block']}")
                        print()
                
                # Summary statistics
                total_containers = len(stats)
                avg_cpu = sum(float(s['CPUPerc'].rstrip('%')) for s in stats if s['CPUPerc']) / max(len(stats), 1)
                
                print("📊 Summary")
                print("-" * 30)
                print(f"Total containers: {total_containers}")
                print(f"Average CPU usage: {avg_cpu:.1f}%")
                
                # Resource warnings
                high_cpu_containers = [s for s in stats if float(s['CPUPerc'].rstrip('%') or 0) > 80]
                if high_cpu_containers:
                    print(f"⚠️  High CPU usage: {len(high_cpu_containers)} containers > 80%")
                
                print("\n" + "=" * 80)
                print("💡 Tips:")
                print("   📋 View logs: python device_manager.py logs [service-name]")
                print("   📈 Scale devices: python device_manager.py scale pi 5")
                print("   🛑 Stop simulation: python device_manager.py stop")
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                print(f"\n{self.reset_color}👋 Monitoring stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(5)
    
    def _get_usage_color(self, usage_pct):
        """Get color code based on usage percentage"""
        if usage_pct > 80:
            return '\033[91m'  # Red
        elif usage_pct > 60:
            return '\033[93m'  # Yellow
        elif usage_pct > 40:
            return '\033[92m'  # Green
        else:
            return '\033[94m'  # Blue
    
    def export_metrics(self, output_file="device_metrics.json"):
        """Export current metrics to JSON file"""
        stats = self.get_container_stats()
        network_count = self.get_network_stats()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'network_containers': network_count,
            'container_stats': stats,
            'summary': {
                'total_containers': len(stats),
                'device_types': {}
            }
        }
        
        # Group statistics by device type
        for stat in stats:
            device_type = self.classify_container(stat['Name'])
            if device_type not in metrics['summary']['device_types']:
                metrics['summary']['device_types'][device_type] = 0
            metrics['summary']['device_types'][device_type] += 1
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📊 Metrics exported to: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description="Monitor federated learning device simulation")
    parser.add_argument('--export', help='Export metrics to JSON file and exit')
    parser.add_argument('--once', action='store_true', help='Show stats once and exit')
    
    args = parser.parse_args()
    
    monitor = DeviceMonitor()
    
    if args.export:
        monitor.export_metrics(args.export)
    elif args.once:
        stats = monitor.get_container_stats()
        if stats:
            print("📊 Current Device Statistics:")
            for stat in stats:
                print(f"  {stat['Name']}: CPU {stat['CPUPerc']}, Memory {stat['MemUsage']}")
        else:
            print("❌ No containers running")
    else:
        monitor.display_dashboard()


if __name__ == "__main__":
    main()