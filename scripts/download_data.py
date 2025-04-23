#!/usr/bin/env python
"""
Download PGLib-OPF dataset for ML-OPF project.
This script downloads the required data files from the web.
"""

import os
import argparse
import requests
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download PGLib-OPF dataset')
    parser.add_argument('--case', type=str, default='case118', 
                        choices=['case5', 'case14', 'case30', 'case57', 'case118', 'case300'],
                        help='Case name (default: case118)')
    parser.add_argument('--data-dir', type=str, default='data', 
                        help='Data directory (default: data)')
    parser.add_argument('--force', action='store_true',
                        help='Force download even if files exist')
    
    return parser.parse_args()

def download_file(url, target_path, force=False):
    """
    Download a file from a URL to a target path.
    
    Args:
        url: URL to download from
        target_path: Path to save the file
        force: Whether to overwrite existing files
    
    Returns:
        True if download successful, False otherwise
    """
    if os.path.exists(target_path) and not force:
        print(f"File already exists: {target_path}")
        return True
    
    try:
        print(f"Downloading {url} to {target_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Download complete: {target_path}")
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main(args):
    """Main function."""
    # Create data directory if it doesn't exist
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # PGLib-OPF base URLs
    pglib_github_url = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master"
    pglib_nrel_url = "https://data.nrel.gov/system/files/177"
    
    # File patterns
    m_file = f"pglib_opf_{args.case}_ieee.m"
    csv_file = f"pglib_opf_{args.case}_ieee.csv"
    
    # For case5, the file name is different
    if args.case == 'case5':
        m_file = "pglib_opf_case5_pjm.m"
        csv_file = "pglib_opf_case5_pjm.csv"
    
    # Target paths
    m_path = data_dir / f"pglib_opf_{args.case}.m"
    csv_path = data_dir / f"pglib_opf_{args.case}.csv"
    
    # Download .m file from GitHub
    success_m = download_file(
        f"{pglib_github_url}/{m_file}",
        m_path,
        args.force
    )
    
    # Download .csv file from NREL
    success_csv = download_file(
        f"{pglib_nrel_url}/{csv_file}",
        csv_path,
        args.force
    )
    
    if success_m and success_csv:
        print(f"\nSuccessfully downloaded data for {args.case}")
        print(f"Files saved to:")
        print(f"  {m_path}")
        print(f"  {csv_path}")
    else:
        print(f"\nFailed to download all files for {args.case}")

if __name__ == '__main__':
    args = parse_args()
    main(args) 