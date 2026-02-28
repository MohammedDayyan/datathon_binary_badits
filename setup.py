"""
Setup script for Image Dehazing Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="image-dehazing-pipeline",
    version="1.0.0",
    author="Dehazing Team",
    author_email="team@dehazing.ai",
    description="Complete Image Dehazing Pipeline for Indian Winter Conditions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dehazing/image-dehazing-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "gpu": [
            "torch>=2.0.0+cu117",
            "torchvision>=0.15.0+cu117",
        ],
    },
    entry_points={
        "console_scripts": [
            "dehaze=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "web": ["*.css", "*.js"],
    },
)
