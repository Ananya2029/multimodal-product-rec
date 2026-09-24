"""Drive the Streamlit app headlessly for ONE model: visit every section, use search and filters.

Run by tests/test_app.py in a fresh process per model (keeps memory low).

    python tests/app_smoke.py gated
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


def main(method: str):
    at = AppTest.from_file(APP, default_timeout=300).run()
    check(at, "start")
    at.sidebar.selectbox[0].set_value(method).run()
    check(at, "select model")
    for page in at.radio(key="page").options:
        print("->", page.encode("ascii", "ignore").decode().strip(), flush=True)
        at.radio(key="page").set_value(page).run()
        check(at, page)
    at.radio(key="page").set_value(at.radio(key="page").options[2]).run()  # text search
    at.text_input[0].set_value("blue jeans for men").run()
    at.sidebar.multiselect[0].set_value(["Women"]).run()
    check(at, "text search + filter")
    at.sidebar.multiselect[1].set_value(["Footwear"]).run()
    at.sidebar.multiselect[0].set_value(["Women", "Men"]).run()
    check(at, "filters")
    print("OK", method)


if __name__ == "__main__":
    main(sys.argv[1])
