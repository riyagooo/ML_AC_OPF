#!/usr/bin/env python
"""
Download the standard IEEE 39-bus New England power system data files.
This script retrieves the MATPOWER data files for case39 and prepares them
for use with our ML-AC-OPF project.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('download_case39')

# URLs and file paths
MATPOWER_ZIP_URL = "https://github.com/MATPOWER/matpower/archive/refs/heads/master.zip"
CASE39_FILENAME = "case39.m"
TARGET_DIR = "data/case39"
TEMP_DIR = "temp_download"

def download_matpower_data():
    """Download MATPOWER data files."""
    logger.info(f"Downloading MATPOWER data from {MATPOWER_ZIP_URL}")
    
    # Create temporary directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Download zip file
    zip_path = os.path.join(TEMP_DIR, "matpower.zip")
    try:
        urllib.request.urlretrieve(MATPOWER_ZIP_URL, zip_path)
    except Exception as e:
        logger.error(f"Failed to download MATPOWER data: {e}")
        return False
    
    # Extract zip file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
        logger.info("MATPOWER data downloaded and extracted successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to extract MATPOWER data: {e}")
        return False

def find_case39_file():
    """Find the case39.m file in the extracted MATPOWER data."""
    logger.info("Looking for case39.m file")
    
    # Walk through the extracted directory
    for root, dirs, files in os.walk(TEMP_DIR):
        if CASE39_FILENAME in files:
            return os.path.join(root, CASE39_FILENAME)
    
    logger.error(f"Could not find {CASE39_FILENAME} in the extracted data")
    return None

def prepare_data_dir():
    """Prepare the data directory."""
    os.makedirs(TARGET_DIR, exist_ok=True)
    logger.info(f"Data directory prepared: {TARGET_DIR}")

def copy_matpower_file(case39_file):
    """Copy the MATPOWER .m file to our data directory."""
    target_path = os.path.join(TARGET_DIR, CASE39_FILENAME)
    try:
        shutil.copy2(case39_file, target_path)
        logger.info(f"Copied {CASE39_FILENAME} to {target_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy MATPOWER file: {e}")
        return False

def cleanup():
    """Clean up temporary files."""
    try:
        shutil.rmtree(TEMP_DIR)
        logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory: {e}")

def main():
    """Main execution function."""
    logger.info("Starting IEEE 39-bus data download and preparation")
    
    # Create data directory
    prepare_data_dir()
    
    # Download MATPOWER data
    if not download_matpower_data():
        logger.error("Failed to download data. Exiting.")
        sys.exit(1)
    
    # Find case39.m file
    case39_file = find_case39_file()
    if not case39_file:
        logger.error("Failed to find case39.m file. Exiting.")
        cleanup()
        sys.exit(1)
    
    # Copy MATPOWER file to data directory
    if not copy_matpower_file(case39_file):
        logger.warning("Failed to copy MATPOWER file, but continuing.")
    
    # Cleanup
    cleanup()
    
    logger.info("IEEE 39-bus data download and preparation completed successfully")
    logger.info(f"Data files are available in the {TARGET_DIR} directory")

if __name__ == "__main__":
    main() 