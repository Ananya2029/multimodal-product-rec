"""Runs the real Streamlit app headlessly for every recommendation system and checks that nothing raises.

    python -m pytest -q tests/test_app.py
"""
import gc
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.recommender import available_systems

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def errors(at):
    return [e.value for e in at.exception]


@pytest.fixture(autouse=True)
def free_models():
    yield
    st.cache_resource.clear()  # unload models between tests (machines with little RAM)
    gc.collect()


def test_comparison_tab_shows_verdict():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, errors(at)
    assert any("Recommended:" in s.value for s in at.success)
    assert len(at.metric) >= 3  # one winner card per task (+ GAN metrics once trained)


@pytest.mark.parametrize("system_name", [s.name for s in available_systems()])
def test_streamlit_app_runs_for_every_system(system_name):
    """Every section of the app with every system, each in a fresh process (frees model memory)."""
    import subprocess
    import sys
    smoke = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_smoke.py")
    r = subprocess.run([sys.executable, smoke, system_name], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout[-2000:], r.stderr[-3000:])


def test_live_shop_search():
    from src.live import load_live_catalog
    if load_live_catalog() is None:
        pytest.skip("live catalog not fetched (python -m src.live)")
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.radio(key="page").set_value("🌐 Internet").run()
    at.text_input(key="live_q").set_value("wrist watch").run()
    assert not at.exception, errors(at)


def test_bad_image_url_shows_error_not_crash():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.radio(key="page").set_value("🌐 Internet").run()
    at.text_input(key="img_url").set_value("http://127.0.0.1/secret.png").run()
    assert not at.exception, errors(at)
    assert any("not allowed" in e.value for e in at.error)
