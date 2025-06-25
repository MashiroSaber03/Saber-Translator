import pytest
import sys
import os
from pathlib import Path


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly set up."""
    
    def test_pytest_is_importable(self):
        """Test that pytest can be imported."""
        import pytest
        assert pytest.__version__
    
    def test_pytest_cov_is_importable(self):
        """Test that pytest-cov plugin is available."""
        import pytest_cov
        assert pytest_cov
    
    def test_pytest_mock_is_importable(self):
        """Test that pytest-mock plugin is available."""
        import pytest_mock
        assert pytest_mock
    
    def test_src_is_in_path(self):
        """Test that src directory is in Python path."""
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
        assert src_path in sys.path
    
    def test_conftest_fixtures_available(self, temp_dir, mock_config, mock_logger):
        """Test that conftest fixtures are available."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert mock_config is not None
        assert mock_logger is not None
    
    def test_temp_file_fixture(self, temp_file):
        """Test the temp_file fixture creates files correctly."""
        test_file = temp_file("test.txt", "test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_markers_are_registered(self, pytestconfig):
        """Test that custom markers are registered."""
        markers = pytestconfig.getini("markers")
        marker_names = [m.split(":")[0] for m in markers]
        assert "unit" in marker_names
        assert "integration" in marker_names
        assert "slow" in marker_names
    
    @pytest.mark.unit
    def test_unit_marker_works(self):
        """Test that unit marker can be used."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Test that integration marker can be used."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Test that slow marker can be used."""
        assert True
    
    def test_coverage_configuration_exists(self):
        """Test that coverage configuration is present in pyproject.toml."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists()
        content = pyproject_path.read_text()
        assert "[tool.coverage.run]" in content
        assert "[tool.coverage.report]" in content
    
    def test_project_structure(self):
        """Test that the expected project structure exists."""
        project_root = Path(__file__).parent.parent
        
        # Check main directories
        assert (project_root / "src").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
        # Check __init__.py files
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "unit" / "__init__.py").exists()
        assert (project_root / "tests" / "integration" / "__init__.py").exists()
    
    def test_mock_fixtures_work(self, mock_session, mock_http_client):
        """Test that mock fixtures work as expected."""
        # Test mock_session
        mock_session.set("key", "value")
        assert mock_session.get("key") == "value"
        
        # Test mock_http_client
        response = mock_http_client.get("http://test.com")
        assert response.status_code == 200
        assert response.json() == {"status": "success"}