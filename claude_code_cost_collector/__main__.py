"""Claude Code Cost Collector package execution entry point.

This module serves as the entry point when executed via `python -m claude_code_cost_collector` command.
It calls the main() function from the main module and exits the program with appropriate exit code.
"""

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
