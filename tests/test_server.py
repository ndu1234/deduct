import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from deduct.extractor import Extraction, TripleOut
from deduct.server import create_app


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.text_stream = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return SimpleNamespace(
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=321, output_tokens=12),
            content=[SimpleNamespace(type="text", text="".join(self.chunks))],
        )


class FakeClient:
    def __init__(self):
        extraction = Extraction(
            add=[TripleOut(subject="user", predicate="lives_in", object="Lagos")], summary="User lives in Lagos."
        )
        parse = lambda **kw: SimpleNamespace(  # noqa: E731
            stop_reason="end_turn", usage=SimpleNamespace(input_tokens=50, output_tokens=10), parsed_output=extraction
        )
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **kw: FakeStream(["Got ", "it."]), parse=parse)
        )


def events(text):
    out = []
    for chunk in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in chunk.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_chat_streams_and_persists(tmp_path):
    app = create_app(str(tmp_path / "m.db"), client=FakeClient())
    http = TestClient(app)

    res = http.post("/api/sessions/abc/chat", json={"message": "I live in Lagos"})
    assert res.status_code == 200
    evs = events(res.text)
    assert [e for e, _ in evs] == ["memory", "delta", "delta", "answer", "saved"]
    assert evs[3][1]["content"] == "Got it." and evs[3][1]["answer_input_tokens"] == 321
    assert evs[4][1]["added"] == [["user", "lives_in", "lagos"]]

    session = http.get("/api/sessions/abc").json()
    assert session["summary"] == "User lives in Lagos."
    assert session["messages"][-1]["turn"]["extraction_tokens"] == 60

    assert http.get("/api/sessions").json()[0]["first_message"] == "I live in Lagos"
    fact = {"subject": "user", "predicate": "lives_in", "object": "lagos"}
    assert http.request("DELETE", "/api/facts", json=fact).status_code == 200
    assert http.get("/api/facts").json() == []
    assert http.post("/api/facts", json=fact).json()["added"] == [["user", "lives_in", "lagos"]]

    assert http.get("/").status_code == 200
    assert http.post("/api/sessions/bad id!/chat", json={"message": "x"}).status_code in (400, 404)
