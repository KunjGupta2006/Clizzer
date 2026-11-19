import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from backend.app.eval.harness import print_evaluation_report

if __name__ == "__main__":
    print_evaluation_report()
