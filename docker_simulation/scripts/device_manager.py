#!/usr/bin/env python3
"""
Federated Learning Device Simulation Manager

Usage:
    python device_manager.py start --pi 3 --mobile 2 --workstation 1
    python device_manager.py stop
    python device_manager.py status
    python device_manager.py logs raspberry-pi
    python device_manager.py scale raspberry-pi 5
"""

import argparse
import subprocess
import time
import os
import sys
import json
from pathlib import Path

class DeviceManager:
    """Manage Docker-based device simulation"""
    
    def __init__(self):
        self.compose_file = Path(__file__).parent.parent / "docker-compose.yml"
        self.env_file = Path(__file__).parent.parent / ".env"
        
    def update_env_file(self, **kwargs):
        """Update .env file with new device counts"""
        env_vars = {}
        
        # Read existing .env file
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value
        
        # Update with new values
        for key, value in kwargs.items():
            env_vars[key] = str(value)
        
        # Write back to .env file
        with open(self.env_file, 'w') as f:
            f.write("# Federated Learning Docker Simulation Configuration\n\n")
            f.write("# === DEVICE COUNTS ===\n")
            f.write(f"PI_COUNT={env_vars.get('PI_COUNT', '2')}\n")
            f.write(f"MOBILE_COUNT={env_vars.get('MOBILE_COUNT', '2')}\n")
            f.write(f"WORKSTATION_COUNT={env_vars.get('WORKSTATION_COUNT', '1')}\n")
            f.write(f"SERVER_COUNT={env_vars.get('SERVER_COUNT', '0')}\n\n")
            f.write("# === SIMULATION MODE ===\n")
            f.write(f"MODE={env_vars.get('MODE', 'dev')}\n\n")
            
            # Write other variables
            other_vars = {k: v for k, v in env_vars.items() 
                         if k not in ['PI_COUNT', 'MOBILE_COUNT', 'WORKSTATION_COUNT', 'SERVER_COUNT', 'MODE']}
            for key, value in other_vars.items():
                f.write(f"{key}={value}\n")
    
    def start_federation(self, pi_count=2, mobile_count=2, workstation_count=1, server_count=0, mode="dev"):
        """Start federated learning simulation with specified device counts"""
        print(f"🚀 Starting Federated Learning Simulation")
        print(f"   📱 Devices: {pi_count} Pi + {mobile_count} Mobile + {workstation_count} Workstation + {server_count} Server")
        print(f"   🔧 Mode: {mode}")
        
        # Update environment variables
        self.update_env_file(
            PI_COUNT=pi_count,
            MOBILE_COUNT=mobile_count, 
            WORKSTATION_COUNT=workstation_count,
            SERVER_COUNT=server_count,
            MODE=mode
        )
        
        # Build images first
        print("🔨 Building Docker images...")
        subprocess.run([
            "docker-compose", "-f", str(self.compose_file),
            "build", "--parallel"
        ], check=True)
        
        # Start services
        print("🏃 Starting services...")
        subprocess.run([
            "docker-compose", "-f", str(self.compose_file),
            "up", "-d", "--scale", f"raspberry-pi={pi_count}",
            "--scale", f"mobile={mobile_count}",
            "--scale", f"workstation={workstation_count}",
            "--scale", f"server-device={server_count}"
        ], check=True)
        
        # Wait for server to be ready
        print("⏳ Waiting for FL server to be ready...")
        self._wait_for_server()
        
        print("✅ Federated Learning simulation started!")
        print(f"   🌐 Server: http://localhost:8080")
        print(f"   📊 Monitoring: http://localhost:9100") 
        self.status()
    
    def stop_federation(self):
        """Stop all federated learning services"""
        print("🛑 Stopping Federated Learning simulation...")
        subprocess.run([
            "docker-compose", "-f", str(self.compose_file),
            "down", "-v"
        ])
        print("✅ Simulation stopped")
    
    def status(self):
        """Show status of all devices"""
        print("\n📊 Device Simulation Status")
        print("=" * 60)
        
        # Get container status
        result = subprocess.run([
            "docker-compose", "-f", str(self.compose_file), "ps"
        ], capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        
        # Count devices by type
        device_counts = self._count_devices()
        print(f"\n🔧 Active Devices:")
        for device_type, count in device_counts.items():
            print(f"   {device_type}: {count} containers")
        
        print(f"\n🌐 Access Points:")
        print(f"   FL Server: http://localhost:8080")
        print(f"   Resource Monitor: http://localhost:9100")
    
    def logs(self, service_name="fl-server", follow=False):
        """Show logs for a specific service"""
        cmd = ["docker-compose", "-f", str(self.compose_file), "logs"]
        if follow:
            cmd.append("-f")
        cmd.append(service_name)
        
        subprocess.run(cmd)
    
    def scale(self, device_type, count):
        """Scale a specific device type to a new count"""
        service_map = {
            'pi': 'raspberry-pi',
            'raspberry-pi': 'raspberry-pi',
            'mobile': 'mobile',
            'workstation': 'workstation',
            'server': 'server-device'
        }
        
        service_name = service_map.get(device_type)
        if not service_name:
            print(f"❌ Unknown device type: {device_type}")
            return
        
        print(f"📈 Scaling {device_type} to {count} instances...")
        subprocess.run([
            "docker-compose", "-f", str(self.compose_file),
            "up", "-d", "--scale", f"{service_name}={count}"
        ])
        
        # Update env file
        env_key = f"{device_type.upper().replace('-', '_')}_COUNT"
        self.update_env_file(**{env_key: count})
        
        print(f"✅ {device_type} scaled to {count} instances")
    
    def monitor(self):
        """Show real-time resource monitoring"""
        print("📊 Real-time Device Monitoring (Ctrl+C to exit)")
        print("=" * 80)
        
        try:
            while True:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("📊 Federated Learning Device Monitor")
                print("=" * 60)
                print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                
                # Show container stats
                result = subprocess.run([
                    "docker", "stats", "--no-stream", "--format",
                    "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
                ], capture_output=True, text=True)
                
                if result.stdout:
                    print(result.stdout)
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
    
    def _wait_for_server(self, timeout=60):
        """Wait for FL server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run([
                    "curl", "-s", "-f", "http://localhost:8080/health"
                ], capture_output=True)
                if result.returncode == 0:
                    return True
            except:
                pass
            
            print("   Waiting for server...")
            time.sleep(5)
        
        print("⚠️  Server startup timeout - check logs with: python device_manager.py logs fl-server")
        return False
    
    def _count_devices(self):
        """Count active devices by type"""
        result = subprocess.run([
            "docker", "ps", "--format", "{{.Names}}"
        ], capture_output=True, text=True)
        
        counts = {
            'Raspberry Pi': 0,
            'Mobile': 0, 
            'Workstation': 0,
            'Server Device': 0,
            'FL Server': 0
        }
        
        for line in result.stdout.split('\n'):
            if 'raspberry-pi' in line:
                counts['Raspberry Pi'] += 1
            elif 'mobile' in line:
                counts['Mobile'] += 1
            elif 'workstation' in line:
                counts['Workstation'] += 1
            elif 'server-device' in line:
                counts['Server Device'] += 1
            elif 'fl-server' in line:
                counts['FL Server'] += 1
        
        return counts


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Device Simulation Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start federated learning simulation')
    start_parser.add_argument('--pi', type=int, default=2, help='Number of Raspberry Pi devices')
    start_parser.add_argument('--mobile', type=int, default=2, help='Number of mobile devices')
    start_parser.add_argument('--workstation', type=int, default=1, help='Number of workstation devices')
    start_parser.add_argument('--server', type=int, default=0, help='Number of server devices')
    start_parser.add_argument('--mode', choices=['dev', 'test', 'prod'], default='dev', help='Simulation mode')
    
    # Stop command
    subparsers.add_parser('stop', help='Stop federated learning simulation')
    
    # Status command
    subparsers.add_parser('status', help='Show simulation status')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Show logs for a service')
    logs_parser.add_argument('service', nargs='?', default='fl-server', help='Service name')
    logs_parser.add_argument('-f', '--follow', action='store_true', help='Follow log output')
    
    # Scale command
    scale_parser = subparsers.add_parser('scale', help='Scale a device type')
    scale_parser.add_argument('device_type', help='Device type to scale (pi, mobile, workstation, server)')
    scale_parser.add_argument('count', type=int, help='New device count')
    
    # Monitor command
    subparsers.add_parser('monitor', help='Show real-time monitoring')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = DeviceManager()
    
    if args.command == 'start':
        manager.start_federation(
            pi_count=args.pi,
            mobile_count=args.mobile, 
            workstation_count=args.workstation,
            server_count=args.server,
            mode=args.mode
        )
    elif args.command == 'stop':
        manager.stop_federation()
    elif args.command == 'status':
        manager.status()
    elif args.command == 'logs':
        manager.logs(args.service, args.follow)
    elif args.command == 'scale':
        manager.scale(args.device_type, args.count)
    elif args.command == 'monitor':
        manager.monitor()


if __name__ == "__main__":
    main()