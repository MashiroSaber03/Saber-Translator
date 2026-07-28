"""Saber Translator backend-first v2 implementation.

Importing this package is intentionally side-effect free.  In particular, it
must not create a Flask application, connect to SQLite/Chroma, discover
plugins, or import local-model dependencies.
"""

__all__ = ["__version__"]

__version__ = "0.1.0-dev"
