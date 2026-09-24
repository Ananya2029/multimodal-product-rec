"""The Streamlit app with every exported from-scratch model, each in a fresh process."""
import os
import subprocess
import sys

import pytest

from src.scratch import serve

METHODS = list(serve.export_info()["methods"]) if serve.available() else []


def test_results_page_loads():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"),
                           default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.skipif(not METHODS, reason="no exported models (run the Colab notebook)")
@pytest.mark.parametrize("method", METHODS)
def test_app_every_section(method):
    smoke = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_smoke.py")
    r = subprocess.run([sys.executable, smoke, method], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout[-2000:], r.stderr[-3000:])
