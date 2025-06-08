"""Setup configuration for Crypto Trading Agent."""

from setuptools import setup, find_packages

# Read version from version.py
with open('src/utils/version.py', 'r') as f:
    exec(f.read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crypto_trading_agent",
    version=__version__,
    author="Biorevtech-Agents",
    author_email="media@biorev.us",
    description="An autonomous AI agent for cryptocurrency trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Biorevtech-Agents/crypto_trading_agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.12,<3.13",
    install_requires=[
        "numpy>=1.24.0,<1.25.0",
        "pandas>=2.0.0,<2.1.0",
        "torch>=2.0.0,<3.0.0",
        "ta-lib>=0.4.0",
        "packaging>=23.0",  # For version parsing
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto_trading_agent=src.main:main",
        ],
    },
) 