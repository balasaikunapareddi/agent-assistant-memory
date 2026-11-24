"""CLI entry point: ``python -m agent_assistant_memory [demo|chat]``."""

from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] == "demo":
        from .demo import main as demo_main
        sys.argv = [sys.argv[0]] + [a for a in args if a != "demo"]
        return demo_main()
    print("usage: python -m agent_assistant_memory demo [--llm stub|gemini] [--budget N]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
