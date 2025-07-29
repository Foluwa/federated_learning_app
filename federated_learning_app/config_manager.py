class ConfigManager:
    """Manages dynamic configuration for different runtime modes"""
    
    def __init__(self, context):
        self.context = context
        self.mode = self._get_mode()
        self.config = self._load_mode_config()
        
    def _get_mode(self):
        """Get the current runtime mode"""
        mode = self.context.run_config.get("mode", "dev").lower()
        if mode not in ["dev", "test", "prod"]:
            print(f"⚠️  Unknown mode '{mode}', defaulting to 'dev'")
            mode = "dev"
        return mode
    
    def _load_mode_config(self):
        """Load configuration based on current mode"""
        mode_configs = {
            "dev": {
                "teacher_epochs": self.context.run_config.get("dev-teacher-epochs", 2),
                "student_epochs": self.context.run_config.get("dev-student-epochs", 2),
                "dataset_size": self.context.run_config.get("dev-dataset-size", 1000),
                "num_server_rounds": self.context.run_config.get("dev-num-server-rounds", 2),
                "fraction_fit": self.context.run_config.get("dev-fraction-fit", 0.5),
                "local_epochs": self.context.run_config.get("dev-local-epochs", 1),
            },
            "test": {
                "teacher_epochs": self.context.run_config.get("test-teacher-epochs", 4),
                "student_epochs": self.context.run_config.get("test-student-epochs", 3),
                "dataset_size": self.context.run_config.get("test-dataset-size", 5000),
                "num_server_rounds": self.context.run_config.get("test-num-server-rounds", 3),
                "fraction_fit": self.context.run_config.get("test-fraction-fit", 0.5),
                "local_epochs": self.context.run_config.get("test-local-epochs", 1),
            },
            "prod": {
                "teacher_epochs": self.context.run_config.get("prod-teacher-epochs", 8),
                "student_epochs": self.context.run_config.get("prod-student-epochs", 5),
                "dataset_size": self.context.run_config.get("prod-dataset-size", -1),  # Full dataset
                "num_server_rounds": self.context.run_config.get("prod-num-server-rounds", 5),
                "fraction_fit": self.context.run_config.get("prod-fraction-fit", 0.3),
                "local_epochs": self.context.run_config.get("prod-local-epochs", 2),
            }
        }
        
        # Get mode-specific config with fallbacks
        config = mode_configs.get(self.mode, mode_configs["dev"]).copy()
        
        # Add common configuration
        config.update({
            "use_cached_models": self.context.run_config.get("use-cached-models", True),
            "force_retrain": self.context.run_config.get("force-retrain", False),
            "cache_dir": self.context.run_config.get("model-cache-dir", "./saved_models"),
            "mode": self.mode
        })
        
        return config
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_cache_suffix(self):
        """Get cache suffix based on mode and parameters"""
        if self.mode == "dev":
            return f"dev_{self.config['dataset_size']}_{self.config['teacher_epochs']}_{self.config['student_epochs']}"
        elif self.mode == "test":
            return f"test_{self.config['dataset_size']}_{self.config['teacher_epochs']}_{self.config['student_epochs']}"
        else:
            return "prod"
    
    def print_config(self):
        """Print current configuration"""
        print(f"Runtime Configuration (Mode: {self.mode.upper()})")
        print("-" * 60)
        
        # Training parameters
        print(f"Training Parameters:")
        print(f"   - Teacher epochs: {self.config['teacher_epochs']}")
        print(f"   - Student epochs: {self.config['student_epochs']}")
        if self.config['dataset_size'] == -1:
            print(f"   - Dataset size: Full dataset")
        else:
            print(f"   - Dataset size: {self.config['dataset_size']:,} samples")
        
        # Federated parameters  
        print(f"Federated Learning:")
        print(f"   - Server rounds: {self.config['num_server_rounds']}")
        print(f"   - Fraction fit: {self.config['fraction_fit']}")
        print(f"   - Local epochs: {self.config['local_epochs']}")
        
        # Caching parameters
        print(f"Caching:")
        print(f"   - Use cached models: {self.config['use_cached_models']}")
        print(f"   - Force retrain: {self.config['force_retrain']}")
        print(f"   - Cache directory: {self.config['cache_dir']}")
        
        # Estimated time
        self._print_estimated_time()
    
    def _print_estimated_time(self):
        """Print estimated runtime based on configuration"""
        print(f"Estimated Runtime:")
        
        if self.config['use_cached_models'] and not self.config['force_retrain']:
            print(f"   - With cache: ~30-60 seconds")
            
        base_time = {
            "dev": "2-3 minutes",
            "test": "5-8 minutes", 
            "prod": "15-25 minutes"
        }
        
        print(f"   - Without cache: ~{base_time.get(self.mode, '5-10 minutes')}")
        
        if self.mode == "dev":
            print(f"Development mode: Optimized for speed!")
        elif self.mode == "test":
            print(f"Test mode: Balanced speed/quality")
        else:
            print(f"Production mode: Full quality training")

