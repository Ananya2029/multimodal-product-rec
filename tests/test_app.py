"""The Streamlit app (results page + proposed-model demo), run headlessly in a fresh process."""
import os
import subprocess
import sys

import pytest

from src.scratch import serve


@pytest.mark.skipif(not serve.available(), reason="no exported models (run the Colab/Kaggle notebook)")
def test_app_pages_and_search_modes():
    smoke = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_smoke.py")
    r = subprocess.run([sys.executable, smoke], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout[-2000:], r.stderr[-3000:])


@pytest.mark.skipif(not serve.available(), reason="no exported models")
def test_proposed_model_answers_all_query_types():
    from PIL import Image
    img = Image.open(serve.catalog().iloc[0]["image_file"])
    for q in (serve.encode_query("gated", text="red dress"), serve.encode_query("gated", image=img),
              serve.encode_query("gated", image=img, text="red")):
        idx, s = serve.top_k(serve.scores("gated", q), 10)
        assert len(idx) == 10
    # text search returns the right kind of product
    idx, _ = serve.top_k(serve.scores("gated", serve.encode_query("gated", text="black handbag for women")), 10)
    assert (serve.catalog().iloc[idx]["articleType"] == "Handbags").mean() >= 0.7
