# deduct

Persistent, token-efficient memory for LLM chats, following the Memori approach
("Memori: A Persistent Memory Layer for Efficient, Context-Aware LLM Agents", Borro et al., Memori Labs, 2026).
Instead of resending raw conversation history, each turn sends only:

- **semantic triples** relevant to the question, e.g. `(user | prefers | dark roast coffee)`
- the **rolling summary** of the current session
- summaries of **related past sessions**
- the last couple of raw turns, for short-term continuity

all capped by a token budget (1,500 by default).

## How a turn works

1. **Retrieve**: BM25 (SQLite FTS5) search over triples and summaries, packed into the budget
   ([retriever.py](src/deduct/retriever.py)).
2. **Answer**: the `<memory>` block goes into the user turn, so the system prompt stays cacheable
   ([agent.py](src/deduct/agent.py)).
3. **Augment**: one structured-output call turns the exchange into new triples, retracted triples
   and an updated summary ([extractor.py](src/deduct/extractor.py)). These are deduplicated and
   stored in SQLite ([store.py](src/deduct/store.py)).

## Usage

```bash
uv sync
export ANTHROPIC_API_KEY=...        # or: ant auth login
uv run deduct chat -v               # -v prints the recalled memory each turn
uv run deduct chat --session work   # resume or name a session
uv run deduct facts                 # dump stored triples
uv run deduct ask "where do I work" # preview retrieval, no API call
uv run pytest
```

Each chat turn prints `memory N tok · input N tok vs ~N full-history`, so you can see
the savings grow as the history gets longer.

```python
from deduct import MemoryAgent, MemoryStore
agent = MemoryAgent(MemoryStore("memory.db"), session_id="demo")
print(agent.chat("I moved to Lagos last month").answer)
```
