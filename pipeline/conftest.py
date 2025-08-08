"""
Pytest configuration for the pipeline tests
"""

import sys
from pathlib import Path

# Add the project root to Python path so imports work
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
