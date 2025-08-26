#!/usr/bin/env python3
"""
Download data from HuggingFace dataset.
This script downloads all data from the pixie dataset, preserving directory structure.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files
import shutil


def download_data(dataset_repo="vlongle/pixie", download_dirs=None, force_download=False, local_dir=None):
    """
    Download data from HuggingFace dataset.
    
    Args:
        dataset_repo (str): HuggingFace dataset repository ID
        download_dirs (list): Specific directories to download (None = all)
        force_download (bool): Force re-download even if files exist
        local_dir (str): Local directory to download to (default: project root)
    """
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if local_dir:
        download_path = Path(local_dir)
    else:
        download_path = project_root
    
    download_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Downloading data from {dataset_repo}...")
    print(f"📂 Download location: {download_path}")
    
    try:
        # List all files in the repository to see what's available
        print("\n🔍 Checking available files...")
        repo_files = list_repo_files(repo_id=dataset_repo, repo_type="dataset")
        
        # Filter out README.md and group by top-level directories
        data_files = [f for f in repo_files if f != "README.md" and not f.startswith(".")] 
        available_dirs = set(f.split('/')[0] for f in data_files if '/' in f)
        
        print(f"📋 Available directories: {sorted(available_dirs)}")
        
        # Determine what to download
        if download_dirs:
            dirs_to_download = [d for d in download_dirs if d in available_dirs]
            if len(dirs_to_download) != len(download_dirs):
                missing = set(download_dirs) - available_dirs
                print(f"⚠️  Warning: These directories not found in dataset: {missing}")
        else:
            dirs_to_download = list(available_dirs)
        
        if not dirs_to_download:
            print("❌ No directories to download!")
            return
            
        print(f"📋 Downloading directories: {dirs_to_download}")
        
        # Create allow/ignore patterns for selective download
        if download_dirs:
            # Only download specific directories
            allow_patterns = []
            for dir_name in dirs_to_download:
                allow_patterns.extend([f"{dir_name}/*", f"{dir_name}/**/*"])
        else:
            # Download everything except README
            allow_patterns = None
            
        ignore_patterns = ["README.md", ".gitattributes"]
        
        # Download using snapshot_download for efficiency
        print("\n📦 Starting download...")
        
        downloaded_path = snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            local_dir=download_path,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            force_download=force_download
        )
        
        print(f"✅ Successfully downloaded to {downloaded_path}")
        
        # Report what was downloaded
        print("\n📊 Download summary:")
        total_size = 0
        total_files = 0
        
        for dir_name in dirs_to_download:
            dir_path = download_path / dir_name
            if dir_path.exists():
                dir_files = list(dir_path.rglob("*"))
                dir_files = [f for f in dir_files if f.is_file()]
                dir_size = sum(f.stat().st_size for f in dir_files)
                
                print(f"  📁 {dir_name}: {len(dir_files)} files ({dir_size / (1024*1024):.1f} MB)")
                total_files += len(dir_files)
                total_size += dir_size
            else:
                print(f"  ❌ {dir_name}: Not found after download")
        
        print(f"\n🎉 Download complete!")
        print(f"📊 Total: {total_files} files ({total_size / (1024*1024):.1f} MB)")
        
    except Exception as e:
        print(f"❌ Failed to download from {dataset_repo}: {e}")
        print("   Make sure the dataset exists and you have access to it")


def main():
    parser = argparse.ArgumentParser(description="Download Pixie data from HuggingFace")
    parser.add_argument(
        "--dataset-repo", 
        default="vlongle/pixie",
        help="HuggingFace dataset repository ID (default: vlongle/pixie)"
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        help="Specific directories to download (default: all available)"
    )
    parser.add_argument(
        "--local-dir",
        help="Local directory to download to (default: project root)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist"
    )
    
    args = parser.parse_args()
    
    download_data(
        dataset_repo=args.dataset_repo, 
        download_dirs=args.dirs,
        force_download=args.force,
        local_dir=args.local_dir
    )


if __name__ == "__main__":
    main()
