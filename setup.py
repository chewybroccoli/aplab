"""aplab package setup"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aplab",
    version="0.1.0",
    author="aplab contributors",
    description="Asset Pricing Laboratory - Python library for asset pricing research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aplab",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "statsmodels>=0.13.0",
    ],
    extras_require={
        "panel": ["linearmodels>=4.27"],
        "excel": ["openpyxl>=3.0.0"],
        "full": ["linearmodels>=4.27", "openpyxl>=3.0.0"],
    },
)
