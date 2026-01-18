# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Note that the Python virtual environment in this repository is managed by `uv` and you should use `uv pip` instead of `pip`

## Project Overview

Rubik's Cube MCP Server - A Model Context Protocol (MCP) server that exposes a Rubik's Cube environment for evaluating LLM sequential reasoning and pattern recognition capabilities.

## Architecture

```
rubiks-mcp/
├── rubiks_mcp/
│   ├── __init__.py
│   └── server.py       # MCP server implementation
├── demo_langchain_agent.py  # Demo GLM agent for 2-move scramble
├── pyproject.toml           # Package configuration
└── .env                    # Private API key (excluded from Git)
```

## MCP Tools

The server exposes 6 tools through the MCP protocol:

| Tool | Purpose | Evaluation Dimension |
|-------|----------|---------------------|
| `create_session` | Initialize cube with optional scramble | Problem baseline |
| `observe` | Query cube state (state/faces/descriptions) | State interpretation |
| `rotate` | Apply moves to cube | Action planning, efficiency |
| `get_metrics` | Retrieve performance metrics | Loop detection, move analysis |
| `get_history` | Get full move sequence | Pattern analysis |
| `check_solution` | Verify if solved | Goal verification |

## Development Commands

### Installation
```bash
# Using uv (recommended)
uv pip install magiccube mcp

# Using pip
pip install magiccube mcp
```

### Running the MCP Server
```bash
python rubiks_mcp/server.py
```

The server uses stdio for MCP protocol communication. It maintains state server-side in a global sessions dictionary, allowing multiple concurrent evaluation runs.

### Running the Demo Agent
```bash
# Ensure .env file contains GLM_API_KEY
python demo_langchain_agent.py
```

The demo agent demonstrates:
- MCP client connection via subprocess JSON-RPC
- GLM-4.5-air model integration for reasoning
- Complete observe-reason-act-verify cycle
- Solution of 2-move scramble (R U → U' R')

## Key Design Decisions

### MCP Tool Semantics
- Tools follow an observe-act-verify cycle rather than providing solution shortcuts
- Sessions maintain state server-side for evaluation isolation
- `observation_mode` parameter controls information granularity (full/faces/partial)
- Move history is tracked individually for loop detection and pattern analysis

### Evaluation Philosophy

Success is not measured solely by solving the cube. The evaluation metrics capture reasoning quality:

| Metric | What It Reveals | Interpretation |
|---------|-----------------|---------------|
| Loop detection | Stuck reasoning | Repeated patterns indicate inability to progress |
| Unique move count | Exploration breadth | Low count suggests local search bias |
| Move distribution | Action preferences | Imbalanced distribution indicates focus on subset of faces |
| Observation count | Information gathering | High count may indicate inefficient state interpretation |
| Time tracking | Computational efficiency | Duration measures actual reasoning time |
| Scramble vs solution | Solution quality | Fewer moves than scramble indicates shortcuts found |

### State Representation

The cube uses SIGN notation for moves. Standard moves:
- Basic: R, L, U, D, F, B (clockwise)
- Inverse: R', L', U', D', F', B' (counter-clockwise)
- Double: R2, L2, U2, D2, F2, B2 (180°)

Cube state is represented as 54-character string for 3x3x3 (6 faces × 9 stickers), mapping to solved state `YYYYYYYYYRRRRRRRRRGGGGGGGGGOOOOOOOOOBBBBBBBBBWWWWWWWWW`.

## Important Notes

- The `.env` file contains private API keys and is **excluded** from Git via `.gitignore`
- The `demo_langchain_agent.py` file contains absolute paths - update server path if relocating
- The magiccube library uses enums (Color, Face) that require string conversion in output
- Session state persists only for the server process lifetime - restarting clears all sessions
- GLM API key must be set in `GLM_API_KEY` environment variable before running demo
- Check the user's operation system before editing files, when editing in WSL2, make sure to use relative path and forward slashes `/`, when you encountered errors like `Failed Writing/Reading File`, please check on these issues
