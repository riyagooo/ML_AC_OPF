#!/usr/bin/env python
"""
ML-OPF: Machine Learning for Optimal Power Flow
Setup script for the ml_opf package
"""

from setuptools import setup, find_packages

setup(
    name="ml_opf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch>=2.0.0",
        "matplotlib",
        "networkx",
        "pyyaml",
        # Optional dependencies:
        # "gurobipy",  # Requires license, not installable via pip
        "pypower",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
        "gnn": [
            "torch-geometric",
            "torch-scatter",
            "torch-sparse",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "ml-opf=ml_opf.__main__:main",
        ],
    },
    author="ML-OPF Team",
    description="Machine Learning for Optimal Power Flow",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)