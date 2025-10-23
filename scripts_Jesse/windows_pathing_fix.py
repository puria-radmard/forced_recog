# Fix Python path to find modules when running scripts from scripts_Jesse directory
import sys
import os
from pathlib import Path

def fix_pathing():
    """
    Fix Python path to ensure modules can be imported when running scripts from scripts_Jesse directory.
    Uses the script's location rather than working directory for more reliable path resolution.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    
    # Add project root to Python path if not already present
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added project root to Python path: {project_root}")
    else:
        print(f"Project root already in Python path: {project_root}")
    
    print(f"Python path updated. First few entries: {sys.path[:3]}")

## Imports and Variables