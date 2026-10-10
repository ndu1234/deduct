from types import SimpleNamespace

from deduct.agent import MemoryAgent
from deduct.extractor import Extraction, TripleOut
from deduct.retriever import Retriever
from deduct.store import MemoryStore, Triple, fts_query


def test_fts_query_drops_stopwords():
    assert fts_query("Where does the user work?") == '"user" OR "work"'
    assert fts_query("the of and") is None


def test_triples_dedupe_and_remove():
    s = MemoryStore(":memory:")
    assert s.add_triples([Triple("User", "lives_in", "Lagos")], "s1") == [Triple("user", "lives_in", "lagos")]
    assert s.add_triples([Triple("user", "lives_in", " lagos ")], "s2") == []  # normalized duplicate
    assert len(s.remove_triples([Triple("user", "lives_in", "lagos")])) == 1
    assert s.search_triples("lagos") == []  # FTS index stays in sync


def test_retrieval_respects_budget_and_relevance():
    s = MemoryStore(":memory:")
    s.add_triples([Triple("user", "prefers", "dark roast coffee"), Triple("user", "works_at", "acme corp")], "s1")
    s.add_triples([Triple(f"project_{i}", "uses", f"library number {i} with a long name") for i in range(200)], "s1")
    s.set_summary("s1", "User planned a coffee tasting trip to Ethiopia.")

    ctx = Retriever(s, budget_tokens=1500).build("what coffee do I like?", session_id="s2")
    assert Triple("user", "prefers", "dark roast coffee") in ctx.triples
    assert Triple("user", "works_at", "acme corp") not in ctx.triples
    assert ctx.related_summaries == ["User planned a coffee tasting trip to Ethiopia."]

    big = Retriever(s, budget_tokens=200).build("library", session_id="s2")
    assert 0 < len(big.triples) < 200 and big.tokens <= 200


class FakeClient:
    """Stands in for anthropic.Anthropic: answers chat calls, returns a canned extraction."""

    def __init__(self, extraction: Extraction):
        self.extraction = extraction
        self.sent = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create, parse=self._parse))

    def _usage(self):
        return SimpleNamespace(input_tokens=100, output_tokens=20)

    def _create(self, **kw):
        self.sent.append(kw)
        return SimpleNamespace(
            stop_reason="end_turn", usage=self._usage(), content=[SimpleNamespace(type="text", text="Noted!")]
        )

    def _parse(self, **kw):
        return SimpleNamespace(stop_reason="end_turn", usage=self._usage(), parsed_output=self.extraction)


def test_agent_round_trip_stores_and_recalls():
    store = MemoryStore(":memory:")
    client = FakeClient(
        Extraction(add=[TripleOut(subject="user", predicate="prefers", object="dark roast coffee")],
                   summary="User shared coffee preference.")
    )
    first = MemoryAgent(store, client=client, session_id="a")
    r = first.chat("I really like dark roast coffee")
    assert r.added == [Triple("user", "prefers", "dark roast coffee")]
    assert store.get_summary("a") == "User shared coffee preference."
    turn = store.session_messages("a")[-1]["turn"]
    assert turn["added"] == [["user", "prefers", "dark roast coffee"]] and turn["answer_input_tokens"] == 100

    # A new session gets the fact back through memory, not through raw history.
    second = MemoryAgent(store, client=client, session_id="b", recent_turns=0)
    second.chat("Which coffee should I buy?")
    last_user = client.sent[-1]["messages"][-1]["content"]
    assert "(user | prefers | dark roast coffee)" in last_user
    assert "I really like dark roast coffee" not in last_user


def test_stemmed_retrieval_matches_predicate_words():
    s = MemoryStore(":memory:")
    s.add_triples([Triple("user", "works_at", "kora"), Triple("user", "moving_to", "nairobi")], "s1")
    found = s.search_triples("Where am I moving and who do I work for?")
    assert set(found) == {Triple("user", "works_at", "kora"), Triple("user", "moving_to", "nairobi")}
