import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    # Fallback when running from source without installed package metadata
    __version__ = "0.0.0"
