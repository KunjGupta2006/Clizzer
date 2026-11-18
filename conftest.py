import sys
from pathlib import Path

# Ensure project root is available to pytest and tests
sys.path.insert(0, str(Path(__file__).resolve().parent))
