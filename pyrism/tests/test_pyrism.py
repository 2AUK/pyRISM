"""
Unit and regression test for the pyrism package.
"""

# Import package, test suite, and other packages as needed
import pyrism
import pytest
import sys

def test_pyrism_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyrism" in sys.modules
