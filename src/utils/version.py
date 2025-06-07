"""Version management utilities for the AI Encryption Agent."""

import sys
import pkg_resources
from typing import Tuple, Optional
from packaging import version

__version__ = "0.1.0"  # Initial version
REQUIRED_PYTHON_VERSION = "3.8"
REQUIRED_PACKAGES = {
    "numpy": ">=1.24.0,<1.25.0",
    "pandas": ">=2.0.0,<2.1.0",
    "torch": ">=2.0.0,<3.0.0",
    "ta-lib": ">=0.4.0",
}

def get_version() -> str:
    """Get the current version of the application."""
    return __version__

def check_python_version() -> Tuple[bool, str]:
    """
    Check if the current Python version meets requirements.
    
    Returns:
        Tuple[bool, str]: (is_compatible, message)
    """
    current_version = '.'.join(map(str, sys.version_info[:2]))
    is_compatible = current_version.startswith(REQUIRED_PYTHON_VERSION)
    
    message = (
        f"Python version check passed. Current version: {current_version}"
        if is_compatible
        else f"Python version {current_version} is not compatible. Required: {REQUIRED_PYTHON_VERSION}"
    )
    
    return is_compatible, message

def check_package_version(package_name: str) -> Tuple[bool, str]:
    """
    Check if a specific package meets version requirements.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        Tuple[bool, str]: (is_compatible, message)
    """
    if package_name not in REQUIRED_PACKAGES:
        return False, f"Package {package_name} is not in the requirements list"
    
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        required_spec = REQUIRED_PACKAGES[package_name]
        
        # Parse the requirement specification
        required = pkg_resources.Requirement.parse(f"{package_name}{required_spec}")
        is_compatible = installed_version in required
        
        message = (
            f"Package {package_name} version {installed_version} is compatible"
            if is_compatible
            else f"Package {package_name} version {installed_version} does not meet requirement: {required_spec}"
        )
        
        return is_compatible, message
    except pkg_resources.DistributionNotFound:
        return False, f"Package {package_name} is not installed"

def check_all_dependencies() -> Tuple[bool, list]:
    """
    Check all required dependencies.
    
    Returns:
        Tuple[bool, list]: (all_compatible, [messages])
    """
    messages = []
    all_compatible = True
    
    # Check Python version
    python_compatible, python_message = check_python_version()
    messages.append(python_message)
    all_compatible &= python_compatible
    
    # Check each required package
    for package in REQUIRED_PACKAGES:
        is_compatible, message = check_package_version(package)
        messages.append(message)
        all_compatible &= is_compatible
    
    return all_compatible, messages

def get_package_version(package_name: str) -> Optional[str]:
    """
    Get the installed version of a package.
    
    Args:
        package_name: Name of the package
        
    Returns:
        Optional[str]: Version string if package is installed, None otherwise
    """
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None 