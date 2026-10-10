"""Interactive chat with persistent memory.

    deduct chat [--db memory.db] [--session ID] [--verbose]
    deduct facts [--db memory.db]
    deduct ask "question" [--db memory.db]     # show what memory would be retrieved, no API call
"""

from __future__ import annotations

import argparse
import sys

from .retriever import Retriever
from .store import MemoryStore


def cmd_chat(args: argparse.Namespace) -> None:
    from .agent import MemoryAgent

    store = MemoryStore(args.db)
    agent = MemoryAgent(
        store, session_id=args.session, model=args.model, effort=args.effort, memory_budget=args.budget
    )
    print(f"session {agent.session_id}  ·  /facts  /stats  /quit")
    totals = {"sent": 0, "full": 0, "extract": 0}
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            break
        if msg == "/facts":
            for t in store.all_triples():
                print(" ", t.render())
            continue
        if msg == "/stats":
            print(" ", store.stats(), totals)
            continue

        r = agent.chat(msg)
        totals["sent"] += r.answer_input_tokens
        totals["full"] += r.full_history_tokens
        totals["extract"] += r.extraction_tokens
        print(f"\nclaude> {r.answer}")
        if args.verbose and r.memory.render():
            print(f"\n[recalled]\n{r.memory.render()}")
        print(
            f"\n[memory {r.memory.tokens} tok · input {r.answer_input_tokens} tok "
            f"vs ~{r.full_history_tokens} full-history · extract {r.extraction_tokens} tok · "
            f"+{len(r.added)}/-{len(r.removed)} facts]"
        )


def cmd_facts(args: argparse.Namespace) -> None:
    store = MemoryStore(args.db)
    for t in store.all_triples():
        print(t.render())
    print(store.stats(), file=sys.stderr)


def cmd_ask(args: argparse.Namespace) -> None:
    ctx = Retriever(MemoryStore(args.db), budget_tokens=args.budget).build(args.query, session_id="")
    print(ctx.render() or "(nothing relevant in memory)")
    print(f"~{ctx.tokens} tokens", file=sys.stderr)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .server import create_app

    app = create_app(args.db, model=args.model, effort=args.effort, memory_budget=args.budget)
    print(f"deduct UI → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def main() -> None:
    p = argparse.ArgumentParser(prog="deduct", description="Token-efficient persistent memory for LLM chats")
    p.add_argument("--db", default="memory.db")
    p.add_argument("--budget", type=int, default=1500, help="max tokens of recalled memory per turn")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat")
    c.add_argument("--session")
    c.add_argument("--model", default="claude-opus-5-5")
    c.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    c.add_argument("-v", "--verbose", action="store_true", help="print the recalled memory block")
    c.set_defaults(fn=cmd_chat)

    w = sub.add_parser("serve", help="run the web UI")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--model", default="claude-opus-5-5")
    w.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    w.set_defaults(fn=cmd_serve)

    sub.add_parser("facts").set_defaults(fn=cmd_facts)

    a = sub.add_parser("ask")
    a.add_argument("query")
    a.set_defaults(fn=cmd_ask)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
