from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_chat_rebuild_and_clear():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=60)
    assert not app.exception
    assert app.chat_input[0].disabled
    app.checkbox[0].check().run()
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value("Is there a lifetime limit?").run(timeout=60)
    assert len(app.chat_message) == 2
    app.button[1].click().run()
    assert len(app.chat_message) == 0
    app.selectbox[0].select("Local Ollama").run()
    assert app.chat_input[0].disabled
    assert len(app.text_input) == 3
