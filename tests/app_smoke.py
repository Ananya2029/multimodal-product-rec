"""Drive the Streamlit app headlessly: both pages, every search mode, filters.

    python tests/app_smoke.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(ROOT, "app.py")


def check(at, where):
    if at.exception:
        print(f"FAIL [{where}]:", [e.value for e in at.exception])
        sys.exit(1)


def main():
    at = AppTest.from_file(APP, default_timeout=300).run()
    check(at, "results page")
    assert len(at.metric) >= 4, "results page should show the proposed model's headline metrics"
    at.radio(key="page").set_value(at.radio(key="page").options[1]).run()
    check(at, "demo page")
    for mode in at.radio(key="mode").options:  # photo modes need an upload: they show the upload prompt
        at.radio(key="mode").set_value(mode).run()
        check(at, mode.encode("ascii", "ignore").decode())
    at.radio(key="mode").set_value(at.radio(key="mode").options[0]).run()
    at.text_input(key="q_text").set_value("blue jeans for men").run()
    at.slider(key="k").set_value(20).run()
    check(at, "text search + number of results")
    at.radio(key="mode").set_value(at.radio(key="mode").options[3]).run()
    seen = {at.selectbox(key="q_item").value}
    for _ in range(3):  # the Random button must actually change the selected product
        at.button[0].click().run()
        check(at, "random product")
        seen.add(at.selectbox(key="q_item").value)
    if len(seen) < 2:
        print("FAIL [random product]: the selected product did not change")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
