import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file in the temp directory."""
    def _create_temp_file(name="test_file.txt", content=""):
        file_path = temp_dir / name
        file_path.write_text(content)
        return file_path
    return _create_temp_file


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    config = MagicMock()
    config.get = Mock(side_effect=lambda key, default=None: {
        "debug": False,
        "log_level": "INFO",
        "api_key": "test_key",
        "max_retries": 3,
        "timeout": 30,
    }.get(key, default))
    return config


@pytest.fixture
def mock_logger():
    """Mock logger object."""
    logger = MagicMock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample image file path."""
    image_path = temp_dir / "sample_image.png"
    # Create a minimal PNG file (1x1 pixel)
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    image_path.write_bytes(png_data)
    return image_path


@pytest.fixture
def sample_text_file(temp_file):
    """Create a sample text file."""
    return temp_file("sample.txt", "This is a test file.\nIt has multiple lines.\n")


@pytest.fixture
def mock_translation_response():
    """Mock translation API response."""
    return {
        "translated_text": "这是一个测试文件。\n它有多行。\n",
        "source_language": "en",
        "target_language": "zh",
        "confidence": 0.95
    }


@pytest.fixture
def mock_ocr_response():
    """Mock OCR API response."""
    return {
        "text": "This is extracted text",
        "bounding_boxes": [
            {"text": "This", "box": [10, 10, 50, 30]},
            {"text": "is", "box": [60, 10, 80, 30]},
            {"text": "extracted", "box": [90, 10, 150, 30]},
            {"text": "text", "box": [160, 10, 200, 30]}
        ],
        "confidence": 0.98
    }


@pytest.fixture
def mock_session():
    """Mock session object."""
    session = MagicMock()
    session.id = "test_session_123"
    session.data = {}
    session.get = Mock(side_effect=lambda key, default=None: session.data.get(key, default))
    session.set = Mock(side_effect=lambda key, value: session.data.update({key: value}))
    return session


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for API calls."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json = Mock(return_value={"status": "success"})
    response.text = "Success"
    client.get = Mock(return_value=response)
    client.post = Mock(return_value=response)
    client.put = Mock(return_value=response)
    client.delete = Mock(return_value=response)
    return client


@pytest.fixture
def mock_flask_app():
    """Mock Flask application."""
    app = MagicMock()
    app.config = {
        "SECRET_KEY": "test_secret",
        "DEBUG": True,
        "TESTING": True
    }
    return app


@pytest.fixture
def env_vars(monkeypatch):
    """Helper fixture to set environment variables."""
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))
    return _set_env


@pytest.fixture
def reset_modules():
    """Reset imported modules to ensure clean imports."""
    modules_to_reset = [m for m in sys.modules.keys() if m.startswith('src.')]
    yield
    for module in modules_to_reset:
        if module in sys.modules:
            del sys.modules[module]


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Clean up any leftover temporary files
    temp_patterns = [
        "/tmp/test_*",
        "/tmp/pytest-*",
    ]
    for pattern in temp_patterns:
        for path in Path("/tmp").glob(pattern.replace("/tmp/", "")):
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)