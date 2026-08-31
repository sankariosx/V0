"""
Run the COMPLETE V0 A+B+C validation using all available CPU cores,
with checkpointing/resume. Works from any directory.

Usage:
    python3 run_v0_parallel.py                # uses cpu_count()-1 workers
    python3 run_v0_parallel.py --workers 8     # override worker count

See v0_lab_final/parallel_runner.py for full documentation of what is and
is not parallelized, the determinism guarantees, and the resume behavior.
"""
import sys
from pathlib import Path

root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from parallel_runner import main

if __name__ == "__main__":
    main()
