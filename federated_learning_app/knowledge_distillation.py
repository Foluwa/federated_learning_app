import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict
import copy
import os
import json
from datetime import datetime

# ============================================================================
# DEVICE TIER CONFIGURATIONS
# ============================================================================

DEVICE_PROFILES = {
    'raspberry_pi': {
        'name': 'Raspberry Pi 4',
        'memory_mb': 2048,
        'target_inference_ms': 100,
        'priority': 'efficiency',
        'conv_channels': [2, 4],  # Very small
        'fc_units': 32,
        'batch_size': 8,
        'description': 'Edge IoT device - maximum efficiency'
    },
    'mobile': {
        'name': 'Mobile Device',
        'memory_mb': 4096,
        'target_inference_ms': 50,
        'priority': 'balanced',
        'conv_channels': [4, 8],  # Small
        'fc_units': 60,
        'batch_size': 16,
        'description': 'Smartphone/tablet - balanced performance'
    },
    'workstation': {
        'name': 'Medical Workstation', 
        'memory_mb': 8192,
        'target_inference_ms': 20,
        'priority': 'accuracy',
        'conv_channels': [8, 16],  # Medium
        'fc_units': 120,
        'batch_size': 32,
        'description': 'Clinical workstation - high accuracy'
    },
    'server': {
        'name': 'Hospital Server',
        'memory_mb': 16384,
        'target_inference_ms': 10,
        'priority': 'accuracy',
        'conv_channels': [16, 32],  # Large
        'fc_units': 256,
        'batch_size': 64,
        'description': 'Data center deployment - maximum accuracy'
    }
}

# ============================================================================
# KNOWLEDGE DISTILLATION COMPONENTS WITH DEVICE TIER SUPPORT
# ============================================================================

class TeacherModel(nn.Module):
    """Teacher model for PathMNIST (adapted for grayscale input)"""
    def __init__(self, num_classes=9):
        super(TeacherModel, self).__init__()
        # Modified for grayscale input (1 channel instead of 3)
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),  # Changed from 3 to 1 input channels
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DeviceSpecificStudentModel(nn.Module):
    """Device-specific student model with configurable architecture"""
    def __init__(self, device_profile, num_classes=9):
        super(DeviceSpecificStudentModel, self).__init__()
        self.device_profile = device_profile
        
        # Extract architecture parameters
        conv_channels = device_profile['conv_channels']
        fc_units = device_profile['fc_units']
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, conv_channels[0], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], 5)
        
        # Calculate flattened size after convolutions
        # Input: 28x28 -> conv1: 24x24 -> pool: 12x12 -> conv2: 8x8 -> pool: 4x4
        flattened_size = conv_channels[1] * 4 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units // 2)
        self.fc3 = nn.Linear(fc_units // 2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_device_info(self):
        """Get device profile information"""
        return {
            'device_type': self.device_profile.get('name', 'Unknown'),
            'memory_mb': self.device_profile.get('memory_mb', 0),
            'target_inference_ms': self.device_profile.get('target_inference_ms', 0),
            'conv_channels': self.device_profile.get('conv_channels', []),
            'fc_units': self.device_profile.get('fc_units', 0)
        }


# Keep original StudentModel for backward compatibility
class StudentModel(DeviceSpecificStudentModel):
    """Original student model - now inherits from device-specific version"""
    def __init__(self, num_classes=9):
        # Use mobile profile as default for backward compatibility
        super(StudentModel, self).__init__(DEVICE_PROFILES['mobile'], num_classes)


def create_student_model(device_type='mobile', num_classes=9):
    """Factory function to create device-specific student models"""
    if device_type not in DEVICE_PROFILES:
        print(f"⚠️  Unknown device type '{device_type}', using 'mobile' as fallback")
        device_type = 'mobile'
    
    profile = DEVICE_PROFILES[device_type]
    model = DeviceSpecificStudentModel(profile, num_classes)
    
    print(f"🔧 Created {profile['name']} model:")
    print(f"   - Conv channels: {profile['conv_channels']}")
    print(f"   - FC units: {profile['fc_units']}")
    print(f"   - Target inference: {profile['target_inference_ms']}ms")
    print(f"   - Memory limit: {profile['memory_mb']}MB")
    
    return model


def get_device_recommendations():
    """Get recommended device types based on capabilities"""
    return {
        'edge_iot': 'raspberry_pi',
        'mobile_app': 'mobile', 
        'clinical_workstation': 'workstation',
        'data_center': 'server'
    }


class ModelCache:
    """Handle saving and loading of trained models with mode awareness"""
    
    def __init__(self, cache_dir="./saved_models", cache_suffix=""):
        self.cache_dir = cache_dir
        self.cache_suffix = cache_suffix
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_model_paths(self):
        """Get paths for teacher and student models with mode suffix"""
        suffix = f"_{self.cache_suffix}" if self.cache_suffix else ""
        return {
            'teacher': os.path.join(self.cache_dir, f'teacher_model{suffix}.pth'),
            'student': os.path.join(self.cache_dir, f'student_model{suffix}.pth'),
            'metadata': os.path.join(self.cache_dir, f'model_metadata{suffix}.json')
        }
    
    def models_exist(self):
        """Check if cached models exist"""
        paths = self.get_model_paths()
        return (os.path.exists(paths['teacher']) and 
                os.path.exists(paths['student']) and 
                os.path.exists(paths['metadata']))
    
    def save_models(self, teacher_model, student_model, metadata=None):
        """Save trained models to disk with enhanced metadata"""
        paths = self.get_model_paths()
        
        print("💾 Saving trained models to cache...")
        
        # Save model state dicts
        torch.save(teacher_model.state_dict(), paths['teacher'])
        torch.save(student_model.state_dict(), paths['student'])
        
        # Save enhanced metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'timestamp': datetime.now().isoformat(),
            'teacher_params': sum(p.numel() for p in teacher_model.parameters()),
            'student_params': sum(p.numel() for p in student_model.parameters()),
            'cache_suffix': self.cache_suffix,
            'model_version': '1.0'
        })
        
        with open(paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Models saved to: {self.cache_dir}")
        print(f"   - Teacher: {os.path.basename(paths['teacher'])}")
        print(f"   - Student: {os.path.basename(paths['student'])}")
        print(f"   - Metadata: {os.path.basename(paths['metadata'])}")
        if self.cache_suffix:
            print(f"   - Cache mode: {self.cache_suffix}")
        
    def load_models(self, device='cpu'):
        """Load cached models from disk"""
        if not self.models_exist():
            raise FileNotFoundError("No cached models found")
            
        paths = self.get_model_paths()
        
        print("📁 Loading cached models...")
        
        # Load metadata
        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)
        
        # Create model instances
        teacher_model = TeacherModel(num_classes=9)
        student_model = StudentModel(num_classes=9)
        
        # Load state dicts
        teacher_model.load_state_dict(torch.load(paths['teacher'], map_location=device))
        student_model.load_state_dict(torch.load(paths['student'], map_location=device))
        
        # Move to device
        teacher_model.to(device)
        student_model.to(device)
        
        print(f"✅ Models loaded from cache (saved: {metadata.get('timestamp', 'unknown')})")
        print(f"   - Teacher: {metadata.get('teacher_params', 'unknown'):,} params")
        print(f"   - Student: {metadata.get('student_params', 'unknown'):,} params")
        if self.cache_suffix:
            print(f"   - Cache mode: {self.cache_suffix}")
        
        return teacher_model, student_model, metadata
    
    def clear_cache(self):
        """Delete cached models"""
        paths = self.get_model_paths()
        for path in paths.values():
            if os.path.exists(path):
                os.remove(path)
        print(f"🗑️  Model cache cleared ({self.cache_suffix})")


class MultiDeviceKnowledgeDistiller:
    """Enhanced distiller that creates models for multiple device tiers"""
    def __init__(self, device='cpu', cache_dir="./saved_models", cache_suffix=""):
        self.device = device
        self.cache_dir = cache_dir
        self.cache_suffix = cache_suffix
        self.device_caches = {}
        
        # Initialize cache for each device type
        for device_type in DEVICE_PROFILES.keys():
            device_cache_suffix = f"{cache_suffix}_{device_type}" if cache_suffix else device_type
            self.device_caches[device_type] = ModelCache(cache_dir, device_cache_suffix)

    def get_or_train_multi_device_models(self, train_loader, config, target_devices=['mobile'], use_cache=True, force_retrain=False):
        """Create models for multiple device types"""
        
        # Extract training parameters from config
        teacher_epochs = config.get('teacher_epochs', 8)
        student_epochs = config.get('student_epochs', 5)
        mode = config.get('mode', 'dev')
        
        print(f"🏭 Multi-Device Model Generation ({mode} mode)")
        print(f"   Target devices: {target_devices}")
        print("-" * 50)
        
        # Step 1: Train or load teacher model
        teacher_model = self._get_or_train_teacher(train_loader, config, use_cache, force_retrain)
        
        # Step 2: Create student models for each target device
        device_models = {}
        
        for device_type in target_devices:
            print(f"\n🔧 Processing {device_type} ({DEVICE_PROFILES[device_type]['name']})...")
            
            # Check device-specific cache
            device_cache = self.device_caches[device_type]
            should_use_cache = (use_cache and not force_retrain and device_cache.models_exist())
            
            if should_use_cache:
                print(f"⚡ Loading cached {device_type} model...")
                try:
                    _, student_model, metadata = device_cache.load_models(self.device)
                    
                    # Verify configuration match
                    cached_config = metadata.get('training_config', {})
                    if (cached_config.get('teacher_epochs') == teacher_epochs and 
                        cached_config.get('student_epochs') == student_epochs and
                        cached_config.get('mode') == mode):
                        device_models[device_type] = student_model
                        print(f"✅ {device_type} model loaded from cache")
                        continue
                    else:
                        print(f"⚠️  Cache mismatch for {device_type}, retraining...")
                except Exception as e:
                    print(f"⚠️  Error loading {device_type} cache: {str(e)}, retraining...")
            
            # Train device-specific student model
            print(f"🎯 Training {device_type} student model...")
            student_model = create_student_model(device_type, num_classes=9)
            
            # Perform knowledge distillation
            student_model = self._distill_device_student(
                teacher_model, student_model, train_loader,
                epochs=student_epochs, device_type=device_type
            )
            
            # Save to device-specific cache
            if use_cache:
                training_config = {
                    'teacher_epochs': teacher_epochs,
                    'student_epochs': student_epochs,
                    'mode': mode,
                    'device_type': device_type,
                    'dataset_size': config.get('dataset_size', 'unknown')
                }
                device_cache.save_models(teacher_model, student_model, {'training_config': training_config})
                print(f"💾 {device_type} model saved to cache")
            
            device_models[device_type] = student_model
        
        # Display summary
        self._print_device_comparison(teacher_model, device_models)
        
        return teacher_model, device_models
    
    def _get_or_train_teacher(self, train_loader, config, use_cache, force_retrain):
        """Get or train teacher model (shared across all devices)"""
        teacher_epochs = config.get('teacher_epochs', 8)
        
        # Use a shared teacher cache
        teacher_cache = ModelCache(self.cache_dir, f"{self.cache_suffix}_teacher" if self.cache_suffix else "teacher")
        
        if use_cache and not force_retrain and teacher_cache.models_exist():
            print("📚 Loading cached teacher model...")
            teacher_model, _, _ = teacher_cache.load_models(self.device)
            return teacher_model
        
        print("🎓 Training teacher model...")
        teacher_model = TeacherModel(num_classes=9)
        teacher_model = self._train_teacher(teacher_model, train_loader, epochs=teacher_epochs)
        
        if use_cache:
            teacher_cache.save_models(teacher_model, teacher_model, {'type': 'teacher'})
        
        return teacher_model
    
    def _train_teacher(self, teacher_model, train_loader, epochs=8, lr=0.001):
        """Train teacher model"""
        print(f"🎓 Training teacher model ({epochs} epochs)...")
        teacher_model.to(self.device)
        teacher_model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(teacher_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                if isinstance(batch, dict):
                    images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                else:
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = teacher_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            print(f"    Teacher Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("✅ Teacher training complete")
        return teacher_model
    
    def _distill_device_student(self, teacher_model, student_model, train_loader, epochs=5, device_type='mobile'):
        """Perform knowledge distillation for specific device"""
        device_profile = DEVICE_PROFILES[device_type]
        
        print(f"🎯 Knowledge distillation for {device_profile['name']} ({epochs} epochs)...")
        
        teacher_model.eval()
        student_model.train()
        student_model.to(self.device)
        
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        temperature = 3.0
        alpha = 0.3
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                if isinstance(batch, dict):
                    images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                else:
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                
                # Teacher forward pass
                with torch.no_grad():
                    teacher_output = teacher_model(images)
                
                # Student forward pass
                student_output = student_model(images)
                
                # Knowledge distillation loss
                soft_targets = F.softmax(teacher_output / temperature, dim=1)
                soft_predictions = F.log_softmax(student_output / temperature, dim=1)
                kd_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)
                
                # Hard target loss
                hard_loss = F.cross_entropy(student_output, labels)
                
                # Combined loss
                loss = alpha * hard_loss + (1 - alpha) * kd_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            print(f"    {device_type} KD Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print(f"✅ {device_type} knowledge distillation complete")
        return student_model
    
    def _print_device_comparison(self, teacher_model, device_models):
        """Print comparison of all device models"""
        print(f"\n📊 Multi-Device Model Comparison")
        print("=" * 80)
        
        # Teacher stats
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        teacher_size = sum(p.numel() * 4 for p in teacher_model.parameters()) / (1024 * 1024)
        
        print(f"🎓 Teacher Model:")
        print(f"   Parameters: {teacher_params:,} | Size: {teacher_size:.2f} MB")
        print()
        
        print(f"🔧 Device-Specific Student Models:")
        print("-" * 80)
        
        for device_type, model in device_models.items():
            profile = DEVICE_PROFILES[device_type]
            params = sum(p.numel() for p in model.parameters())
            size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
            compression = params / teacher_params
            
            print(f"{profile['name']:20} | {params:>8,} params | {size:>6.2f} MB | "
                  f"{compression:>6.1%} | {profile['target_inference_ms']:>3}ms target")
        
        print("-" * 80)


# Keep original KnowledgeDistiller for backward compatibility
class KnowledgeDistiller(MultiDeviceKnowledgeDistiller):
    """Original knowledge distiller - now uses multi-device version"""
    
    def get_or_train_models(self, train_loader, config, use_cache=True, force_retrain=False):
        """Original interface - creates single mobile device model"""
        teacher_model, device_models = self.get_or_train_multi_device_models(
            train_loader, config, target_devices=['mobile'], use_cache=use_cache, force_retrain=force_retrain
        )
        return teacher_model, device_models['mobile']

    def get_model_info(self, model):
        """Get model size and parameter count"""
        param_count = sum(p.numel() for p in model.parameters())
        model_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # MB
        return param_count, model_size


# ============================================================================
# UTILITY FUNCTIONS FOR CACHE MANAGEMENT
# ============================================================================

def check_cache_status(cache_dir="./saved_models"):
    """Check status of all cached models"""
    print("📋 Cache Status Report")
    print("-" * 50)
    
    if not os.path.exists(cache_dir):
        print("❌ Cache directory does not exist")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    
    if not cache_files:
        print("❌ No cached models found")
        return
    
    for metadata_file in cache_files:
        try:
            with open(os.path.join(cache_dir, metadata_file), 'r') as f:
                metadata = json.load(f)
            
            mode = metadata.get('cache_suffix', 'unknown')
            timestamp = metadata.get('timestamp', 'unknown')
            config = metadata.get('training_config', {})
            
            print(f"✅ Cached model: {mode}")
            print(f"   - Saved: {timestamp}")
            print(f"   - Teacher epochs: {config.get('teacher_epochs', 'unknown')}")
            print(f"   - Student epochs: {config.get('student_epochs', 'unknown')}")
            print(f"   - Dataset size: {config.get('dataset_size', 'unknown')}")
            print()
            
        except Exception as e:
            print(f"⚠️  Error reading {metadata_file}: {str(e)}")


def clear_all_cache(cache_dir="./saved_models"):
    """Clear all cached models"""
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("🗑️  All cache files cleared")
    else:
        print("❌ Cache directory does not exist")


def clear_mode_cache(mode, cache_dir="./saved_models"):
    """Clear cache for specific mode"""
    cache = ModelCache(cache_dir, mode)
    cache.clear_cache()


# ============================================================================
# CONVENIENCE FUNCTIONS FOR TESTING
# ============================================================================

# if __name__ == "__main__":
#     # Example usage for testing
#     print("🔧 Knowledge Distillation Module Loaded")
#     print("Available functions:")
#     print("  - check_cache_status()")
#     print("  - clear_all_cache()")
#     print("  - clear_mode_cache('dev')")
    
#     # Check current cache status
#     check_cache_status()