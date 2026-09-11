# Agent Guidelines for decision-lab (dlab)

This project is a Python CLI harness for agentic data science that runs opencode in automated Docker-sandboxed environments with parallel subagents.

## Development Environment & Testing

- Virtual Environment: Use [venv/](venv/) or local Python environment (Python 3.10+ required). Activate with `source .venv/bin/activate`.
- Install Dev Dependencies: `uv pip install -e ".[dev]"`
- Run All Tests: `pytest tests/ -v`
- Run Single Test File: `pytest tests/test_config.py -v` (e.g. [tests/test_config.py](tests/test_config.py))
- Testing Rules:
  - NEVER use mocks — tests must use real implementations.
  - NEVER skip tests — report failures clearly.
  - Do NOT set `PYTESTER_FLAGS="cxx="` (C++ compilation is faster for PyTensor deterministics).

## Core CLI Usage

- Basic Run: `dlab run --dpack <dpack> --data <data-dir> --env-file .env --prompt "..."`
- Interactive Wizards: `dlab create-dpack`, `dlab create-parallel-agent`
- Live Monitoring: `dlab connect <work-dir>`, `dlab timeline <work-dir>`, `dlab view <work-dir>`
- For full CLI details, see [docs/cli-reference.md](docs/cli-reference.md) and [skills/dlab-cli/SKILL.md](skills/dlab-cli/SKILL.md).

## Coding Style & Conventions

- Formatting: Manual PEP 8 formatting (do NOT run Black or Ruff).
- File Naming: `snake_case` (e.g. `my_module.py`).
- Imports: Always at top of file, sorted with `isort` (stdlib, third-party, local).
- Type Annotations: Full type annotations on all functions and variables.
- Documentation: NumPy-style docstrings with Parameters/Returns sections.
- Error Handling: Use built-in exceptions (`ValueError`, `TypeError`, etc.) with descriptive messages. NEVER wrap code in generic `try/except Exception: pass` or `except: pass`.
- Architectural Patterns: Prefer simple functions over classes/singletons/abstractions unless necessary.

## Documentation References

- Architecture & Standards: [.claude/CLAUDE.md](.claude/CLAUDE.md)
- Decision Packs: [docs/decision-packs.md](docs/decision-packs.md)
- Parallel Subagents: [docs/parallel-agents.md](docs/parallel-agents.md)
- Session Management: [docs/sessions.md](docs/sessions.md)
- Docker Integration: [docs/docker.md](docs/docker.md)
- Log Processing: [docs/log-processing.md](docs/log-processing.md)
- Project Overview: [README.md](README.md)
