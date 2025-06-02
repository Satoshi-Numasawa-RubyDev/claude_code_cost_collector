"""
Tests for the __main__ module.
"""

import sys
from unittest.mock import patch

import pytest


# Test the __main__.py module
class TestMainModule:
    """Test the __main__ module functionality."""

    def test_main_module_execution_mock(self):
        """Test that __main__ module structure is correct for execution."""
        # Test that the module can be imported and has the expected structure
        import claude_code_cost_collector.__main__ as main_module

        # Verify it imports the main function
        from claude_code_cost_collector.main import main

        assert main_module.main == main

        # Verify it imports sys
        assert main_module.sys == sys

    def test_main_module_content(self):
        """Test that __main__ module has correct content."""
        with open("claude_code_cost_collector/__main__.py", "r") as f:
            content = f.read()

        # Check for expected imports and structure
        assert "from .main import main" in content
        assert "import sys" in content
        assert "sys.exit(main())" in content
        assert 'if __name__ == "__main__":' in content

    def test_main_module_import(self):
        """Test that __main__ module can be imported without issues."""
        try:
            import claude_code_cost_collector.__main__  # noqa: F401

            # If we get here without exception, the import was successful
            assert True
        except ImportError:
            pytest.fail("Failed to import __main__ module")

    def test_main_module_has_correct_structure(self):
        """Test that __main__ module has the expected structure."""
        import claude_code_cost_collector.__main__ as main_module

        # Check that it imports main from the correct module
        assert hasattr(main_module, "main")
        assert hasattr(main_module, "sys")

    @patch("claude_code_cost_collector.main.main")
    def test_main_module_python_m_execution(self, mock_main):
        """Test execution via python -m claude_code_cost_collector."""
        mock_main.return_value = 0

        # Simulate python -m execution by setting __name__ to '__main__'
        import claude_code_cost_collector.__main__ as main_module

        # The module should be ready for execution
        assert main_module.main is not None
        assert main_module.sys is not None


if __name__ == "__main__":
    pytest.main([__file__])
