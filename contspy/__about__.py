try:
    # Python 3.8
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

try:
    __author__ = metadata.metadata("contspy")["Author"]
except Exception:
    __author__ = "unknown"

try:
    __email__ = metadata.metadata("contspy")["Author-email"]
except Exception:
    __email__ = "unknown"

try:
    __license__ = metadata.metadata("contspy")["License"]
except Exception:
    __license__ = "unknown"

try:
    __status__ = metadata.metadata("contspy")["Classifier"]
except Exception:
    __status__ = "unknown"

try:
    __version__ = metadata.version("contspy")
except Exception:
    __version__ = "unknown"
