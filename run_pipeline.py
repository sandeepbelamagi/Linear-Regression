"""
run_pipeline.py — End-to-End Pipeline Runner
==============================================
Runs all 7 phases in order. Use this as the entry point.

Usage:
    python run_pipeline.py              # run all phases
    python run_pipeline.py --phase 4    # run specific phase
    python run_pipeline.py --phase 2 3  # run phases 2 and 3

Each phase is independent and can be run separately.
Outputs are saved to outputs/ folder.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

MPLCONFIGDIR = ROOT / ".tmp" / "mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_phase1():
    print("\n" + "=" * 60)
    print("PHASE 1: Exploratory Data Analysis")
    print("=" * 60)
    import runpy
    runpy.run_path(str(ROOT / "src" / "eda.py"), run_name="__main__")


def run_phase2():
    print("\n" + "=" * 60)
    print("PHASE 2: Preprocessing Pipeline")
    print("=" * 60)
    import runpy
    runpy.run_path(str(ROOT / "src" / "preprocessing" / "pipeline.py"), run_name="__main__")


def run_phase3():
    print("\n" + "=" * 60)
    print("PHASE 3: Feature Engineering")
    print("=" * 60)
    import runpy
    runpy.run_path(str(ROOT / "src" / "features" / "engineer.py"), run_name="__main__")


def run_phase4():
    print("\n" + "=" * 60)
    print("PHASE 4: Model Training & Comparison")
    print("=" * 60)
    import runpy
    runpy.run_path(str(ROOT / "src" / "models" / "train.py"), run_name="__main__")


def run_phase5():
    print("\n" + "=" * 60)
    print("PHASE 5: Uncertainty Estimation & Routing")
    print("=" * 60)
    import runpy
    runpy.run_path(str(ROOT / "src" / "models" / "uncertainty.py"), run_name="__main__")


def run_phase6():
    print("\n" + "=" * 60)
    print("PHASE 6: Drift Detection & Monitoring")
    print("=" * 60)
    import runpy
    runpy.run_path(str(ROOT / "src" / "monitoring" / "drift.py"), run_name="__main__")


def run_phase7():
    print("\n" + "=" * 60)
    print("PHASE 7: Production API")
    print("=" * 60)
    print("[info] FastAPI server code is at src/api/serve.py")
    print("[info] To start the server, run:")
    print("         uvicorn src.api.serve:app --host 0.0.0.0 --port 8000 --reload")
    print("[info] API docs available at: http://localhost:8000/docs")


PHASES = {
    1: ("EDA",                     run_phase1),
    2: ("Preprocessing Pipeline",  run_phase2),
    3: ("Feature Engineering",     run_phase3),
    4: ("Model Training",          run_phase4),
    5: ("Uncertainty & Routing",   run_phase5),
    6: ("Drift & Monitoring",      run_phase6),
    7: ("Production API",          run_phase7),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sales forecasting pipeline phases.")
    parser.add_argument("--phase", nargs="+", type=int,
                        choices=list(PHASES.keys()),
                        help="Phase(s) to run (default: all)")
    args = parser.parse_args()

    phases_to_run = args.phase if args.phase else list(PHASES.keys())

    print("\nSales Forecasting Production Pipeline")
    print(f"   Running phases: {phases_to_run}")

    total_start = time.time()

    for phase_num in phases_to_run:
        name, fn = PHASES[phase_num]
        phase_start = time.time()
        try:
            fn()
            elapsed = round(time.time() - phase_start, 1)
            print(f"\n[ok] Phase {phase_num} ({name}) complete in {elapsed}s")
        except Exception as e:
            print(f"\n[error] Phase {phase_num} ({name}) failed: {e}")
            import traceback
            traceback.print_exc()

    total = round(time.time() - total_start, 1)
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {total}s")
    print(f"Outputs saved to: {ROOT / 'outputs'}")
    print("=" * 60)
