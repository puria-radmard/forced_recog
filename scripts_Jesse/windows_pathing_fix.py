# Fix working directory and Python path to find src module from scripts directory
import sys
import os

def fix_pathing():
    # Get current working directory and manually remove "scripts" if present
    cwd = os.getcwd()
    print(f"Original working directory: {cwd}")

    # If we're in scripts directory, change to the parent directory
    if cwd.endswith("scripts_Jesse"):
        project_root = os.path.dirname(cwd)
        os.chdir(project_root)
        print(f"Changed working directory to: {os.getcwd()}")
    else:
        project_root = cwd
        print(f"Already in project root: {project_root}")

    # Add project root to Python path if not already present
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"Final working directory: {os.getcwd()}")
    print(f"Python path updated. First few entries: {sys.path[:3]}")

## Imports and Variables