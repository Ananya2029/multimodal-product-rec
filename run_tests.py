"""Run the test suite one file per process (so model memory is released between files).

    python run_tests.py

Plain `python -m pytest` also works on machines with plenty of RAM; on a ~7 GB laptop that is
also running other software, loading CLIP/SigLIP many times in one process can exhaust memory.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
files = sorted((ROOT / "tests").glob("test_*.py"))
failed = []
t0 = time.time()
for f in files:
    print(f"\n=== {f.name} ===", flush=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(f.relative_to(ROOT))], cwd=ROOT)
    if r.returncode not in (0, 5):  # 5 = no tests collected
        failed.append(f.name)
print(f"\n{'ALL TEST FILES PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}  ({(time.time() - t0) / 60:.1f} min)")
sys.exit(1 if failed else 0)
