import talib
import torch
import pandas as pd
import numpy as np
import transformers
import nltk
import textblob
import matplotlib.pyplot as plt
import seaborn as sns
import ai_encryp_agent  # Will automatically check environment

print("Python packages imported successfully!")
print(f"TA-Lib version: {talib.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# Get current version
from ai_encryp_agent import __version__
print(f"Running version: {__version__}")

# Manual environment check
from ai_encryp_agent import check_environment
check_environment()  # Will exit if environment is incompatible 