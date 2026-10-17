"""Launch from any directory without depending on editable .pth discovery."""

from memory_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
