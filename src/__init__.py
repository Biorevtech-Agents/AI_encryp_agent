"""AI Encryption Agent main package initialization."""

import sys
from .utils.version import check_all_dependencies, get_version

__version__ = get_version()

def check_environment():
    """
    Check if the current environment meets all requirements.
    Raises SystemExit if requirements are not met.
    """
    compatible, messages = check_all_dependencies()
    
    # Print all compatibility messages
    for message in messages:
        print(message)
    
    if not compatible:
        print("\nEnvironment check failed. Please ensure all requirements are met.")
        sys.exit(1)
    
    print("\nEnvironment check passed successfully!")

# Run environment check on import
check_environment() 