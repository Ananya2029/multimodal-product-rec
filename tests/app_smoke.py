"""Drive the Streamlit app headlessly for ONE system: visit every section, use search and filters.

Run by tests/test_app.py in a fresh process per system, so memory from loading large models is fully
released between systems (the machine this was built on has 7 GB of RAM).

    python tests/app_smoke.py "CLIP image + text"
"""
import os
import sys

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(ROOT, "app.py")


def check(at, where):
    if at.exception:
        print(f"FAIL [{where}]:", [e.value for e in at.exception])
        sys.exit(1)


def main(system_name: str):
    at = AppTest.from_file(APP, default_timeout=300).run()
    check(at, "start")
    at.sidebar.selectbox[0].set_value(system_name).run()
    check(at, "select system")
    pages = at.radio(key="page").options
    for page in pages:  # every section of the app, with this system selected
        print("->", page.encode("ascii", "ignore").decode().strip(), flush=True)
        at.radio(key="page").set_value(page).run()
        check(at, page)
    # text search, filters (including a combination that matches nothing), weight slider
    at.radio(key="page").set_value(pages[2]).run()
    at.text_input[0].set_value("blue denim jeans for men").run()
    at.sidebar.multiselect[0].set_value(["Women"]).run()
    check(at, "text search + filter")
    if len(at.sidebar.slider) == 2:  # multimodal systems have the weight slider
        at.sidebar.slider[0].set_value(0.8).run()
        check(at, "weight slider")
    at.sidebar.multiselect[1].set_value(["Footwear"]).run()
    at.sidebar.multiselect[0].set_value(["Women", "Men"]).run()
    check(at, "filters")
    print("OK", system_name)


if __name__ == "__main__":
    main(sys.argv[1])
