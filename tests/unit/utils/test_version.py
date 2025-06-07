"""Tests for version management utilities."""

import sys
import pytest
from unittest.mock import patch
from src.utils.version import (
    get_version,
    check_python_version,
    check_package_version,
    check_all_dependencies,
    get_package_version,
    REQUIRED_PYTHON_VERSION,
    REQUIRED_PACKAGES,
)

def test_get_version():
    """Test getting application version."""
    version = get_version()
    assert isinstance(version, str)
    assert version.count('.') == 2  # Ensure semantic versioning format

@pytest.mark.parametrize("version_info,expected", [
    ((3, 8, 0), (True, "Python version check passed. Current version: 3.8")),
    ((3, 9, 0), (False, "Python version 3.9 is not compatible. Required: 3.8")),
    ((3, 7, 0), (False, "Python version 3.7 is not compatible. Required: 3.8")),
])
def test_check_python_version(version_info, expected):
    """Test Python version compatibility checking."""
    with patch('sys.version_info', version_info):
        result = check_python_version()
        assert result == expected

@pytest.mark.parametrize("package_name,version,expected_compatible", [
    ("numpy", "1.24.2", True),
    ("numpy", "1.25.0", False),
    ("pandas", "2.0.1", True),
    ("pandas", "1.5.0", False),
    ("invalid-package", "1.0.0", False),
])
def test_check_package_version(package_name, version, expected_compatible):
    """Test package version compatibility checking."""
    with patch('pkg_resources.get_distribution') as mock_dist:
        if package_name in REQUIRED_PACKAGES:
            mock_dist.return_value.version = version
            compatible, message = check_package_version(package_name)
            assert compatible == expected_compatible
            if compatible:
                assert "is compatible" in message
            else:
                assert "does not meet requirement" in message
        else:
            compatible, message = check_package_version(package_name)
            assert not compatible
            assert "not in the requirements list" in message

def test_check_all_dependencies():
    """Test checking all dependencies."""
    with patch('sys.version_info', (3, 8, 0)):
        with patch('pkg_resources.get_distribution') as mock_dist:
            # Mock all packages as compatible
            mock_dist.return_value.version = "1.24.2"  # A compatible version
            
            compatible, messages = check_all_dependencies()
            assert compatible
            assert len(messages) == len(REQUIRED_PACKAGES) + 1  # +1 for Python version
            assert all("compatible" in msg for msg in messages)

def test_get_package_version():
    """Test getting package version."""
    with patch('pkg_resources.get_distribution') as mock_dist:
        # Test existing package
        mock_dist.return_value.version = "1.0.0"
        version = get_package_version("existing-package")
        assert version == "1.0.0"
        
        # Test non-existing package
        mock_dist.side_effect = pkg_resources.DistributionNotFound()
        version = get_package_version("non-existing-package")
        assert version is None 