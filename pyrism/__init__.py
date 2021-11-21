"""
pyrism
A pedagogical implementation of the RISM equations
"""

# Add imports here
from .rism_ctrl import *
from .Closures import *

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
