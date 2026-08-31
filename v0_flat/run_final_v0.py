import sys
from pathlib import Path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
for p in [str(parent_dir), str(current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)
from tests import run_all_tests
if __name__ == "__main__":
    print("Starting Final V0 Full Validation (A->B->C) with N=60000, TOP30, MAX50, purified scoring")
    print("This will take 12-24 hours on i5-1235U / 16GB - run overnight plugged in")
    report = run_all_tests()
    print("\nDone. Check results/v0_report.json")
