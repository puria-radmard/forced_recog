import modal
import os
from pathlib import Path

def download_volume_files(volume, target_dir, allowed_extensions=('.txt', '.csv')):
    """
    Recursively download files from Modal volume to local directory.
    
    Args:
        volume: Modal volume object
        target_dir: Local target directory path (string or Path)
        allowed_extensions: Tuple of file extensions to download
    """
    target_dir = Path(target_dir)
    
    def download_recursive(volume_path):
        """Recursively traverse and download files from volume path."""
        try:
            entries = volume.listdir(volume_path)
        except Exception as e:
            print(f"Error listing directory {volume_path}: {e}")
            return
        
        for entry in entries:
            # Get the relative path from the volume root
            relative_path = entry.path
            local_path = target_dir / relative_path
            
            if entry.type.name == 'DIRECTORY':
                # Create local directory if it doesn't exist
                local_path.mkdir(parents=True, exist_ok=True)
                print(f"Created/ensured directory: {local_path}")
                
                # Recurse into subdirectory
                download_recursive(f"/{entry.path}")
                
            elif entry.type.name == 'FILE':
                # Check if file has allowed extension
                if any(entry.path.lower().endswith(ext) for ext in allowed_extensions):
                    # Create parent directories if they don't exist
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    # Download file content
                    print(f"Downloading: {entry.path}")
                    data = b""
                    for chunk in volume.read_file(entry.path):
                        data += chunk
                    
                    # Write to local file
                    with open(local_path, 'bw') as f:
                        f.write(data)
                    
                    print(f"Saved: {local_path}")
                        
                else:
                    print(f"Skipping (not allowed extension): {entry.path}")
    
    # Start recursive download from root
    download_recursive('/')

# Usage example
if __name__ == "__main__":
    # Initialize your volume
    results_volume = modal.Volume.from_name("results-vol", create_if_missing=False)
    
    # Set target directory (change this to your desired local path)
    target_directory = "results_and_data/modal_results"
    
    # Download files
    print(f"Starting download to: {os.path.abspath(target_directory)}")
    download_volume_files(results_volume, target_directory)
    print("Download complete!")
