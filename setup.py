from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-opf",
    version="0.1.0",
    author="ML-OPF Team",
    author_email="your.email@example.com",
    description="Machine Learning for Optimal Power Flow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ML-OPF-Project",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/ML-OPF-Project/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "pypower>=5.1.0",
        "networkx>=2.6.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
        ],
        "visualization": [
            "plotly>=5.10.0",
        ],
        "gnn": [
            "torch-geometric>=2.0.0"
        ]
    },
) 