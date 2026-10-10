"""deduct: token-efficient persistent memory via semantic triples + conversation summaries."""

from .retriever import MemoryContext, Retriever
from .store import MemoryStore, Triple

__all__ = ["MemoryAgent", "MemoryContext", "MemoryStore", "Retriever", "Triple"]


def __getattr__(name: str):
    # Lazy so the store/retriever work without the anthropic SDK configured.
    if name == "MemoryAgent":
        from .agent import MemoryAgent

        return MemoryAgent
    raise AttributeError(name)
