"""
PwC Agentic Document Processing — Streamlit Entry Point

Thin entry point that delegates to the UI layer.
"""

import sys
from pathlib import Path

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.pages import render_main_page

render_main_page()
