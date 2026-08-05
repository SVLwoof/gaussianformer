"""Backwards-compat shim. The pipeline now lives in run_scene.py; this module
preserves the `python -m data_external.run_tomatoes` entry point used by older
SLURM scripts like runs/eval_v10b_tomatoes.sh.
"""
from data_external.run_scene import main

if __name__ == "__main__":
    main(scene="tomatoes")
