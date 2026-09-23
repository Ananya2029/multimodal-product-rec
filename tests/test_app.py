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

from src.recommender import SYSTEMS

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
    assert len(at.metric) == 3  # one winner card per task


@pytest.mark.parametrize("system_name", [s.name for s in SYSTEMS])
def test_streamlit_app_runs_for_every_system(system_name):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, errors(at)
    at.sidebar.selectbox[0].set_value(system_name).run()
    assert not at.exception, errors(at)
    # change query product, text query and filters
    at.text_input[0].set_value("blue denim jeans for men").run()
    assert not at.exception, errors(at)
    at.sidebar.multiselect[0].set_value(["Women"]).run()
    assert not at.exception, errors(at)
    if len(at.sidebar.slider) == 2:  # multimodal systems have the weight slider
        at.sidebar.slider[0].set_value(0.8).run()
        assert not at.exception, errors(at)
    # filters that match nothing must not crash
    at.sidebar.multiselect[1].set_value(["Footwear"]).run()
    at.sidebar.multiselect[0].set_value(["Women", "Men"]).run()
    assert not at.exception, errors(at)
