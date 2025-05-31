"""Setup script for Time Series Diffusion Framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r") as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith("#")]

setup(
    name="time-series-diffusion",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular framework for time series diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/time-series-diffusion",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "isort>=5.10.0",
        ],
        "financial": [
            "yfinance>=0.1.70",
            "ta>=0.10.0",
        ],
        "acceleration": [
            "einops>=0.4.0",
            "numba>=0.55.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tsdiff-train=cli.train:main",
            "tsdiff-evaluate=cli.evaluate:main",
            "tsdiff-predict=cli.predict:main",
            "tsdiff-compare=cli.compare:main",
        ],
    },
)