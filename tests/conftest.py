"""Pytest configuration — ensure the project root is on sys.path."""
import sys
from pathlib import Path

# Add project root so that `from src.xxx import ...` works without installing
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
