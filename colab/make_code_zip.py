"""Package the project code for Google Colab -> colab/mmrec_code.zip (upload it in the notebook).

    python colab/make_code_zip.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "colab" / "mmrec_code.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted((ROOT / "src").rglob("*.py")):
        if "__pycache__" not in f.parts:
            z.write(f, f.relative_to(ROOT))
    z.write(ROOT / "requirements.txt", "requirements.txt")
print(f"{out} ({out.stat().st_size / 1024:.0f} KB)")
