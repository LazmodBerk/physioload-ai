"""Execute all PhysioLoad AI notebooks and save outputs inplace."""

import sys
import asyncio
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

success_count = 0
failed_count = 0

for nb_path in sorted(NB_DIR.glob("*.ipynb")):
    print(f"Calistiriliyor: {nb_path.name} ...")
    try:
        nb = nbformat.read(nb_path, as_version=4)
        ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        nbformat.write(nb, nb_path)
        print(f"  -> BASARILI: {nb_path.name}")
        success_count += 1
    except Exception as e:
        print(f"  -> HATA ({nb_path.name}): {e}")
        failed_count += 1

print(f"\nSonuc: {success_count} notebook basariyla calistirildi, {failed_count} hata.")

