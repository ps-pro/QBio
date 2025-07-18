# -*- coding: utf-8 -*-
"""
Comprehensive logging system for quantum DNA analysis
"""

import json
import sys
import platform
from datetime import datetime
from ..config.settings import CONFIG, ANALYSIS_SEEDS, CPU_COUNT, MAX_WORKERS

try:
    import qiskit
    import numpy as np
    import pandas as pd
except ImportError:
    pass

class ComprehensiveLogger:
    """Academic-level logging system for reproducibility"""

    def __init__(self):
        self.start_time = datetime.now()
        self.log_data = {
            'timestamp': self.start_time.isoformat(),
            'configuration': CONFIG.copy(),
            'seeds': ANALYSIS_SEEDS.copy(),
            'environment': self._get_environment_info(),
            'performance': {},
            'statistical_results': {},
            'warnings': [],
            'errors': []
        }

    def _get_environment_info(self):
        """Collect environment information for reproducibility"""
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'cpu_count': CPU_COUNT,
            'max_workers': MAX_WORKERS
        }
        
        # Add library versions if available
        try:
            import qiskit
            env_info['qiskit_version'] = qiskit.__version__
        except ImportError:
            env_info['qiskit_version'] = "not installed"
            
        try:
            import numpy as np
            env_info['numpy_version'] = np.__version__
        except ImportError:
            env_info['numpy_version'] = "not installed"
            
        try:
            import pandas as pd
            env_info['pandas_version'] = pd.__version__
        except ImportError:
            env_info['pandas_version'] = "not installed"

        return env_info

    def log_performance(self, analysis_type, metrics):
        """Log performance metrics"""
        self.log_data['performance'][analysis_type] = metrics

    def log_statistical_result(self, test_name, result):
        """Log statistical test results"""
        self.log_data['statistical_results'][test_name] = result

    def log_warning(self, message):
        """Log warning message"""
        self.log_data['warnings'].append({
            'timestamp': datetime.now().isoformat(),
            'message': message
        })

    def log_error(self, message, exception=None):
        """Log error message"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        if exception:
            error_entry['exception'] = str(exception)
        self.log_data['errors'].append(error_entry)

    def save_log(self, filename='execution_log.json'):
        """Save comprehensive log to file"""
        self.log_data['execution_time'] = str(datetime.now() - self.start_time)

        with open(f'results/logs/{filename}', 'w', encoding='utf-16') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"📋 Comprehensive log saved to results/logs/{filename}")

# Global logger instance
logger = ComprehensiveLogger()