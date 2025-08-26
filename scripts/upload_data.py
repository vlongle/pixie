#!/usr/bin/env python3
"""
Upload data directories to HuggingFace dataset.
This script uploads specified directories to the pixie dataset, preserving their structure.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login
import fnmatch


# Default directories to upload - can be overridden via command line
DEFAULT_UPLOAD_DIRS = [
    "checkpoints_continuous_mse",
    "checkpoints_discrete", 
    "real_scene_data",
    "real_scene_models"
]

# Files to ignore during upload
IGNORE_PATTERNS = [
    "*.pyc", "__pycache__", ".DS_Store", "*.tmp", "*.log",
    "wandb", ".git", ".gitignore"
]


def should_ignore_file(file_path):
    """Check if a file should be ignored based on patterns."""
    file_name = file_path.name
    for pattern in IGNORE_PATTERNS:
        if fnmatch.fnmatch(file_name, pattern):
            return True
    return False


def upload_directory(api, local_dir, repo_id, repo_dir_name=None):
    """
    Upload a directory to HuggingFace dataset using efficient bulk upload.
    
    Args:
        api: HuggingFace API instance
        local_dir (Path): Local directory to upload
        repo_id (str): Repository ID
        repo_dir_name (str): Name in repo (defaults to local dir name)
    """
    if repo_dir_name is None:
        repo_dir_name = local_dir.name
        
    print(f"\n📁 Uploading directory: {local_dir} -> {repo_dir_name}/")
    
    # Count files and calculate size for reporting
    total_files = 0
    total_size = 0
    
    for file_path in local_dir.rglob("*"):
        if file_path.is_file() and not should_ignore_file(file_path):
            total_files += 1
            total_size += file_path.stat().st_size
    
    print(f"  📊 Found {total_files} files ({total_size / (1024*1024):.1f} MB) to upload")
    
    if total_files == 0:
        print("  ⚠️  No files to upload in this directory")
        return 0, 0
    
    try:
        # Use upload_folder for efficient bulk upload
        print(f"  ⏳ Uploading folder (this may take a while for large datasets)...")
        
        # Create ignore patterns list combining our patterns
        ignore_patterns = []
        for pattern in IGNORE_PATTERNS:
            # Convert simple patterns to work with upload_folder
            if pattern == "__pycache__":
                ignore_patterns.extend(["**/__pycache__/**", "__pycache__"])
            elif pattern == ".git":
                ignore_patterns.extend(["**/.git/**", ".git"])
            elif pattern == "wandb":
                ignore_patterns.extend(["**/wandb/**", "wandb"])
            else:
                ignore_patterns.append(pattern)
        
        # Upload the entire folder at once
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=repo_dir_name,
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=ignore_patterns
        )
        
        print(f"  ✅ Successfully uploaded {total_files} files ({total_size / (1024*1024):.1f} MB)")
        
    except Exception as e:
        print(f"  ❌ Failed to upload directory: {e}")
        return 0, 0
    
    return total_files, total_size


def upload_data(dataset_repo="vlongle/pixie", upload_dirs=None, token=None):
    """
    Upload data directories to HuggingFace dataset.
    
    Args:
        dataset_repo (str): HuggingFace dataset repository ID
        upload_dirs (list): List of directory names to upload
        token (str): HuggingFace token (optional, will prompt if not provided)
    """
    
    if upload_dirs is None:
        upload_dirs = DEFAULT_UPLOAD_DIRS
    
    # Login to HuggingFace
    if token:
        login(token=token)
    else:
        print("Please login to HuggingFace Hub...")
        login()
    
    api = HfApi()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"🚀 Uploading data to {dataset_repo}...")
    print(f"📂 Project root: {project_root}")
    print(f"📋 Directories to upload: {upload_dirs}")
    
    total_uploaded_files = 0
    total_uploaded_size = 0
    
    for dir_name in upload_dirs:
        local_dir = project_root / dir_name
        
        if not local_dir.exists():
            print(f"⚠️  Warning: {local_dir} does not exist, skipping...")
            continue
            
        if not local_dir.is_dir():
            print(f"⚠️  Warning: {local_dir} is not a directory, skipping...")
            continue
            
        try:
            files, size = upload_directory(api, local_dir, dataset_repo)
            total_uploaded_files += files
            total_uploaded_size += size
        except Exception as e:
            print(f"❌ Failed to upload directory {dir_name}: {e}")
    
    # Upload README for the dataset
    readme_content = f"""# Pixie Dataset

This dataset contains data and pre-trained models for the Pixie project.

## Contents

{chr(10).join([f"- `{dir_name}/`: {_get_dir_description(dir_name)}" for dir_name in upload_dirs if (project_root / dir_name).exists()])}

## Usage

Use the download script in the Pixie repository to automatically download this data:

```bash
python scripts/download_data.py
```

"""
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=dataset_repo,
            repo_type="dataset",
        )
        print("✅ Successfully uploaded dataset README")
    except Exception as e:
        print(f"❌ Failed to upload README: {e}")
    
    print(f"\n🎉 Upload complete!")
    print(f"📊 Total: {total_uploaded_files} files ({total_uploaded_size / (1024*1024):.1f} MB)")
    print(f"🌐 Visit: https://huggingface.co/datasets/{dataset_repo}")


def _get_dir_description(dir_name):
    """Get description for directory based on its name."""
    descriptions = {
        "checkpoints_continuous_mse": "Continuous material property prediction model checkpoints",
        "checkpoints_discrete": "Discrete material classification model checkpoints",
        "real_scene_data": "Real scene data for evaluation",
        "real_scene_models": "Trained models for real scenes"
    }
    return descriptions.get(dir_name, "Data directory")


def main():
    parser = argparse.ArgumentParser(description="Upload Pixie data to HuggingFace")
    parser.add_argument(
        "--dataset-repo", 
        default="vlongle/pixie",
        help="HuggingFace dataset repository ID (default: vlongle/pixie)"
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        help=f"Directories to upload (default: {DEFAULT_UPLOAD_DIRS})"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace token (optional, will prompt if not provided)"
    )
    
    args = parser.parse_args()
    
    upload_dirs = args.dirs if args.dirs else DEFAULT_UPLOAD_DIRS
    upload_data(dataset_repo=args.dataset_repo, upload_dirs=upload_dirs, token=args.token)


if __name__ == "__main__":
    main()
