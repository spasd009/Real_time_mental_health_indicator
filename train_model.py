from __future__ import annotations

import os
import runpy
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "typing_mental_health_model"
os.chdir(PROJECT_DIR)
runpy.run_path(str(PROJECT_DIR / "train_model.py"), run_name="__main__")