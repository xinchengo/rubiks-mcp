#!/usr/bin/env python3
"""
Rubik's Cube MCP Server

Intent: Provide a deterministic, observable environment for evaluating LLM sequential
reasoning and pattern recognition capabilities. The Rubik's Cube presents a combinatorial
state space (43 quintillion configurations for 3x3x3) where efficient solving requires:
- Multi-step forward planning
- Pattern recognition and abstraction
- Memory of previous states
- Backtracking and error recovery

MCP Semantics: Tools follow an observe-act-verify cycle. Sessions maintain state
server-side, enabling evaluation of LLM reasoning through multiple interaction rounds.

Evaluation Philosophy: Success is not just solving, but how the LLM arrives at the
solution. Loop detection indicates failure to progress; unique move count indicates
exploration diversity; observation count indicates information gathering strategy.
"""

import asyncio
import time
import uuid
from collections import Counter
from typing import Any

from magiccube import Cube, Face, Color
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Server instance
app = Server("rubiks-cube")


class CubeSession:
    """
    Encapsulates a single reasoning trial.

    Intent: Isolate each LLM interaction to prevent state contamination between
    evaluation runs. Each session is an independent problem instance.

    Tracked attributes support evaluation of:
    - Loop detection (indicates stuck reasoning)
    - Time efficiency (indicates computational efficiency)
    - Observation patterns (indicates information gathering strategy)
    """

    def __init__(
        self,
        size: int = 3,
        initial_state: str | None = None,
        scramble: str | None = None,
        scramble_depth: int = 0,
        observation_mode: str = "full",
    ):
        self.session_id = str(uuid.uuid4())
        self.size = size
        self.observation_mode = observation_mode

        # Create the cube
        if initial_state:
            self.cube = Cube(size, initial_state)
        else:
            self.cube = Cube(size)

        # Apply scramble if provided
        self.scramble_applied = None
        self.scramble_moves_count = 0
        if scramble:
            self.cube.rotate(scramble)
            self.scramble_applied = scramble
            self.scramble_moves_count = len(scramble.split()) if scramble else 0
        elif scramble_depth > 0:
            # Generate random scramble
            scramble = self.cube.generate_random_moves(scramble_depth)
            self.cube.rotate(scramble)
            self.scramble_applied = scramble
            self.scramble_moves_count = len(scramble.split()) if scramble else 0

        # Track moves made during the session
        # Evaluation: Full history enables loop detection and pattern analysis
        self.moves: list[str] = []
        self.move_count = 0

        # Time tracking
        # Evaluation: Time between first and last move measures active reasoning duration
        self.created_at = time.time()
        self.first_move_at = None
        self.last_move_at = None

        # State observation tracking
        # Evaluation: High observation count may indicate inefficient information gathering
        self.observation_count = 0


# Global sessions storage
sessions: dict[str, CubeSession] = {}


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    MCP tool registry.

    Intent: Expose a minimal yet complete API for cube interaction. Tools are
    designed to test specific cognitive capabilities rather than provide convenient shortcuts.

    Evaluation mapping:
    - create_session: Establishes problem instance, records baseline
    - observe: Tests state interpretation and abstraction ability
    - rotate: Tests action planning and execution
    - get_metrics: Provides aggregate evaluation data
    - get_history: Enables fine-grained reasoning analysis
    - check_solution: Tests goal verification capability
    """
    return [
        Tool(
            name="create_session",
            description=(
                "Creates a new Rubik's Cube session. Returns a session ID and initial state. "
                "You can start with a solved cube or apply a scramble."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "size": {
                        "type": "integer",
                        "description": "Cube size (NxNxN). Default: 3.",
                        "default": 3,
                        "minimum": 2,
                    },
                    "scramble": {
                        "type": "string",
                        "description": "Scramble moves in SIGN notation (e.g., 'R U R' U''). If omitted, cube starts solved.",
                    },
                    "scramble_depth": {
                        "type": "integer",
                        "description": "Generate random scramble of this length. Ignored if scramble is provided.",
                        "default": 0,
                        "minimum": 0,
                    },
                    "observation": {
                        "type": "string",
                        "enum": ["full", "faces", "partial"],
                        "description": "Observation mode: full (complete state), faces (6 face colors), partial (limited info). Default: full.",
                        "default": "full",
                    },
                },
            },
        ),
        Tool(
            name="observe",
            description=(
                "Returns current cube state. Use this after creating a session or making moves "
                "to see the result. Different formats available: state string, face arrays, descriptions, or 3D ASCII grid."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier from create_session.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["state", "faces", "descriptions", "grid"],
                        "description": "Output format. 'state' = color string, 'faces' = 6 face arrays, 'descriptions' = natural language, 'grid' = 3D ASCII visualization. Default: state.",
                        "default": "state",
                    },
                    "include_move_count": {
                        "type": "boolean",
                        "description": "Show number of moves made so far. Default: true.",
                        "default": True,
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="rotate",
            description=(
                "Applies rotation moves to the cube. Use this for single moves or small sequences. "
                "Supports SIGN notation including advanced moves like R2, U', Fw, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier from create_session.",
                    },
                    "moves": {
                        "type": "string",
                        "description": "Moves in SIGN notation (e.g., 'R L2 U D' F').",
                    },
                },
                "required": ["session_id", "moves"],
            },
        ),
        Tool(
            name="get_metrics",
            description=(
                "Returns performance metrics for session: move count, time elapsed, "
                "loop patterns detected, unique moves, and efficiency indicators."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier from create_session.",
                    },
                    "include_loop_analysis": {
                        "type": "boolean",
                        "description": "Analyze history for repeated patterns and loops. Default: true.",
                        "default": True,
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="get_history",
            description=(
                "Returns the sequence of moves applied to the cube. Useful for analyzing "
                "reasoning patterns and detecting repetitions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier from create_session.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["list", "string", "summary"],
                        "description": "Output format. 'list' = array of moves, 'string' = space-separated, 'summary' = statistics. Default: list.",
                        "default": "list",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent moves to return. 0 = all moves. Default: 0.",
                        "default": 0,
                        "minimum": 0,
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="check_solution",
            description=(
                "Checks if the cube is currently solved. Returns a simple yes/no result "
                "and optionally details about remaining unsolved pieces."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier from create_session.",
                    },
                    "show_details": {
                        "type": "boolean",
                        "description": "Show which pieces are still unsolved. Default: false.",
                        "default": False,
                    },
                },
                "required": ["session_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """
    MCP tool dispatcher.

    Intent: Route tool calls to appropriate handlers. This is the MCP protocol
    entry point for all tool invocations.

    Semantics: Returns TextContent containing JSON-encoded results. The LLM receives
    this as tool output, which it can use to inform subsequent actions.
    """
    if name == "create_session":
        return await handle_create_session(arguments)
    elif name == "observe":
        return await handle_observe(arguments)
    elif name == "rotate":
        return await handle_rotate(arguments)
    elif name == "get_metrics":
        return await handle_get_metrics(arguments)
    elif name == "get_history":
        return await handle_get_history(arguments)
    elif name == "check_solution":
        return await handle_check_solution(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def handle_create_session(args: dict[str, Any]) -> list[TextContent]:
    """
    Initialize a new evaluation session.

    Intent: Create a fresh problem instance with known initial conditions.
    The scramble establishes a known distance to solution.

    Evaluation: The scramble complexity (move count) sets a lower bound on
    solution quality. LLMs that solve in fewer moves than scramble length
    have found shortcuts or used reverse moves.
    """
    size = args.get("size", 3)
    scramble = args.get("scramble")
    scramble_depth = args.get("scramble_depth", 0)
    observation = args.get("observation", "full")

    session = CubeSession(
        size=size,
        scramble=scramble,
        scramble_depth=scramble_depth,
        observation_mode=observation,
    )

    sessions[session.session_id] = session

    # Get current state
    state = session.cube.get()

    result = {
        "session_id": session.session_id,
        "size": session.size,
        "state": state,
        "scramble_applied": session.scramble_applied,
        "observation_mode": session.observation_mode,
        "is_solved": session.cube.is_done(),
    }

    return [TextContent(type="text", text=str(result))]


async def handle_observe(args: dict[str, Any]) -> list[TextContent]:
    """
    Return cube state in requested format.

    Intent: Provide state information at varying abstraction levels to test
    the LLM's ability to interpret and abstract the cube configuration.

    Evaluation:
    - `format="state"`: Tests raw state processing
    - `format="faces"`: Tests face-level abstraction
    - `format="descriptions"`: Tests natural language interpretation
    - Observation count: High values suggest inefficient information gathering

    Semantics: Observations are side-effect free but increment a counter for
    evaluation of information gathering strategy.
    """
    session_id = args["session_id"]
    format_type = args.get("format", "state")
    include_move_count = args.get("include_move_count", True)

    if session_id not in sessions:
        raise ValueError(f"Session not found: {session_id}")

    session = sessions[session_id]
    session.observation_count += 1

    result: dict[str, Any] = {
        "session_id": session_id,
    }

    # Color enum to character mapping
    color_to_char = {
        Color.Y: "Y",
        Color.R: "R",
        Color.G: "G",
        Color.O: "O",
        Color.B: "B",
        Color.W: "W",
    }

    color_names = {"Y": "yellow", "R": "red", "G": "green", "O": "orange", "B": "blue", "W": "white"}
    face_names = {Face.U: "Up", Face.D: "Down", Face.F: "Front", Face.B: "Back", Face.L: "Left", Face.R: "Right"}
    face_short_names = {Face.U: "U", Face.D: "D", Face.F: "F", Face.B: "B", Face.L: "L", Face.R: "R"}

    if format_type == "state":
        result["state"] = session.cube.get()
        result["is_solved"] = session.cube.is_done()
    elif format_type == "faces":
        all_faces = session.cube.get_all_faces()
        # Convert to string representation
        faces_str = {}
        for face_enum, face_array in all_faces.items():
            faces_str[face_short_names[face_enum]] = [
                color_to_char[color] for row in face_array for color in row
            ]
        result["faces"] = faces_str
        result["is_solved"] = session.cube.is_done()
    elif format_type == "descriptions":
        all_faces = session.cube.get_all_faces()

        descriptions = []
        for face_enum, face_array in all_faces.items():
            # Flatten 2D array and convert to colors
            colors = [color_names.get(color_to_char[color], "unknown") for row in face_array for color in row]
            # Determine the dominant color (should be uniform for a solved face)
            dominant_color = max(set(colors), key=colors.count) if colors else "unknown"
            descriptions.append(f"The {face_names[face_enum]} face is {dominant_color}")

        result["description"] = ". ".join(descriptions) + "."
        result["is_solved"] = session.cube.is_done()
    elif format_type == "grid":
        # Return 3D ASCII grid layout matching magiccube's str() output without ANSI codes
        # Map faces for grid layout order: U, then (F,R,B,L), then D
        # This matches the unfolded cube visualization
        all_faces = session.cube.get_all_faces()

        def _format_face_row(face_enum: Face, row_idx: int) -> str:
            """Format a single row of a face."""
            row = all_faces.get(face_enum)[row_idx]
            chars = [color_to_char[color] for color in row]
            return " ".join(chars)

        # Build grid layout
        # U face (centered)
        u_row1 = _format_face_row(Face.U, 0)
        u_row2 = _format_face_row(Face.U, 1)
        u_row3 = _format_face_row(Face.U, 2)
        u_face = f"         {u_row1}\n         {u_row2}\n         {u_row3}\n"

        # Middle faces (F, R, B, L side by side)
        f_row1 = _format_face_row(Face.F, 0)
        f_row2 = _format_face_row(Face.F, 1)
        f_row3 = _format_face_row(Face.F, 2)
        r_row1 = _format_face_row(Face.R, 0)
        r_row2 = _format_face_row(Face.R, 1)
        r_row3 = _format_face_row(Face.R, 2)
        b_row1 = _format_face_row(Face.B, 0)
        b_row2 = _format_face_row(Face.B, 1)
        b_row3 = _format_face_row(Face.B, 2)
        l_row1 = _format_face_row(Face.L, 0)
        l_row2 = _format_face_row(Face.L, 1)
        l_row3 = _format_face_row(Face.L, 2)

        middle_face = (
            f"         {f_row1}         {r_row1}\n"
            f"         {f_row2}         {r_row2}\n"
            f"         {f_row3}         {r_row3}\n"
            f"         {b_row1}         {l_row1}\n"
            f"         {b_row2}         {l_row2}\n"
            f"         {b_row3}         {l_row3}\n"
        )

        # D face (centered)
        d_row1 = _format_face_row(Face.D, 0)
        d_row2 = _format_face_row(Face.D, 1)
        d_row3 = _format_face_row(Face.D, 2)
        d_face = f"         {d_row1}\n         {d_row2}\n         {d_row3}\n"

        grid = u_face + middle_face + d_face
        result["grid"] = grid
        result["is_solved"] = session.cube.is_done()

    if include_move_count:
        result["move_count"] = session.move_count

    return [TextContent(type="text", text=str(result))]


async def handle_rotate(args: dict[str, Any]) -> list[TextContent]:
    """
    Apply one or more moves to the cube.

    Intent: Enable state modification through actions. The LLM must plan
    sequences of moves and observe results.

    Evaluation:
    - Move count: Total moves measures solution efficiency
    - Move distribution: Imbalanced distribution may indicate local search bias
    - Time tracking: Measures computational efficiency of reasoning

    Semantics: Moves are deterministic and irreversible (undo requires explicit inverse).
    Each move in a sequence is tracked individually for loop detection.
    """
    session_id = args["session_id"]
    moves = args["moves"]

    if session_id not in sessions:
        raise ValueError(f"Session not found: {session_id}")

    session = sessions[session_id]

    # Track first move time
    if session.first_move_at is None:
        session.first_move_at = time.time()

    # Parse moves (split by whitespace)
    move_list = moves.strip().split()

    # Apply each move
    for move in move_list:
        session.cube.rotate(move)
        session.moves.append(move)
        session.move_count += 1

    # Update last move time
    session.last_move_at = time.time()

    result = {
        "session_id": session_id,
        "moves_applied": move_list,
        "move_count": session.move_count,
        "state": session.cube.get(),
        "is_solved": session.cube.is_done(),
    }

    return [TextContent(type="text", text=str(result))]


def _analyze_loops(moves: list[str]) -> dict[str, Any]:
    """
    Detect repetitive patterns in move history.

    Intent: Identify when reasoning is stuck in local cycles. Looping
    indicates failure to progress or recognize previously visited states.

    Evaluation:
    - has_loops: Presence of any repeated sequence indicates stalled reasoning
    - loop_patterns: Frequency and position of repetitions
    - longest_repeating_sequence: Largest pattern that repeats

    Design: Detects sequences of length 2-6. Shorter patterns are noise;
    longer patterns indicate deliberate but unproductive repetition.
    """
    if len(moves) < 2:
        return {
            "has_loops": False,
            "loop_patterns": [],
            "longest_repeating_sequence": None,
        }

    loop_analysis = {
        "has_loops": False,
        "loop_patterns": [],
        "longest_repeating_sequence": None,
    }

    # Find repeated individual moves
    move_counts = Counter(moves)

    # Find repeated sequences of length 2-6
    sequence_patterns: dict[tuple[str, ...], int] = {}
    sequence_positions: dict[tuple[str, ...], list[int]] = {}

    for seq_len in range(2, min(7, len(moves))):
        for i in range(len(moves) - seq_len + 1):
            seq = tuple(moves[i : i + seq_len])
            sequence_patterns[seq] = sequence_patterns.get(seq, 0) + 1
            if seq not in sequence_positions:
                sequence_positions[seq] = []
            sequence_positions[seq].append(i)

    # Find patterns that repeat 2+ times
    for seq, count in sequence_patterns.items():
        if count >= 2:
            loop_analysis["has_loops"] = True
            loop_analysis["loop_patterns"].append({
                "pattern": " ".join(seq),
                "occurrences": count,
                "positions": sequence_positions[seq][:5],  # First 5 positions
            })

    # Sort by occurrences descending
    loop_analysis["loop_patterns"].sort(key=lambda x: x["occurrences"], reverse=True)

    # Find longest repeating sequence
    if sequence_patterns:
        # Sort by length first, then by occurrences
        sorted_seqs = sorted(
            sequence_patterns.items(),
            key=lambda x: (len(x[0]), x[1]),
            reverse=True,
        )
        if sorted_seqs and sorted_seqs[0][1] >= 2:
            longest_seq = sorted_seqs[0][0]
            loop_analysis["longest_repeating_sequence"] = {
                "sequence": " ".join(longest_seq),
                "length": len(longest_seq),
                "occurrences": sorted_seqs[0][1],
            }

    return loop_analysis


async def handle_get_metrics(args: dict[str, Any]) -> list[TextContent]:
    """
    Return aggregate evaluation metrics.

    Intent: Provide a comprehensive summary of reasoning quality without
    requiring analysis of raw move history.

    Evaluation dimensions:
    - Efficiency: Move count relative to scramble complexity
    - Diversity: Unique moves indicate exploration breadth
    - Progress: No loops, move distribution balanced
    - Time: Duration measures computational efficiency
    - Information: Observation count indicates gathering strategy

    Semantics: Metrics are computed from tracked session state. This is a
    read-only operation used after solving completes.
    """
    session_id = args["session_id"]
    include_loop_analysis = args.get("include_loop_analysis", True)

    if session_id not in sessions:
        raise ValueError(f"Session not found: {session_id}")

    session = sessions[session_id]

    # Calculate unique moves
    unique_moves = len(set(session.moves))
    move_counter = Counter(session.moves)

    # Time tracking
    current_time = time.time()
    elapsed_time = session.last_move_at - session.first_move_at if session.first_move_at else 0
    total_session_duration = current_time - session.created_at

    # Loop analysis
    loop_result = _analyze_loops(session.moves) if include_loop_analysis and session.moves else {
        "has_loops": False,
        "loop_patterns": [],
        "longest_repeating_sequence": None,
    }

    result = {
        "session_id": session_id,
        "total_moves": session.move_count,
        "unique_moves": unique_moves,
        "move_distribution": dict(move_counter.most_common()),
        "time_tracking": {
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_display": f"{elapsed_time:.1f}s" if elapsed_time else "N/A",
            "total_session_duration_seconds": total_session_duration,
            "average_time_per_move": f"{elapsed_time / session.move_count:.2f}s" if session.move_count > 0 else "N/A",
        },
        "loop_analysis": {
            "has_loops": loop_result["has_loops"],
            "loops_detected": len(loop_result["loop_patterns"]),
            "loop_patterns": loop_result["loop_patterns"][:5],  # Top 5 patterns
            "longest_repeating_sequence": loop_result["longest_repeating_sequence"],
        },
        "observation_count": session.observation_count,
        "scramble_moves": session.scramble_moves_count,
    }

    return [TextContent(type="text", text=str(result))]


async def handle_get_history(args: dict[str, Any]) -> list[TextContent]:
    """
    Return the move history.

    Intent: Enable fine-grained analysis of reasoning strategies. The history
    reveals patterns, dead ends, and decision points.

    Evaluation:
    - Raw list: Detects repeating sequences manually
    - Summary: Shows move distribution and diversity
    - String format: Useful for visualization or external analysis

    Semantics: Returns all moves by default; limit parameter enables
    incremental retrieval for streaming analysis.
    """
    session_id = args["session_id"]
    format_type = args.get("format", "list")
    limit = args.get("limit", 0)

    if session_id not in sessions:
        raise ValueError(f"Session not found: {session_id}")

    session = sessions[session_id]

    # Get moves (apply limit)
    moves = session.moves
    if limit > 0:
        moves = moves[-limit:]

    result: dict[str, Any] = {"session_id": session_id}

    if format_type == "list":
        result["moves"] = moves
        result["count"] = len(moves)
    elif format_type == "string":
        result["moves_string"] = " ".join(moves)
        result["count"] = len(moves)
    elif format_type == "summary":
        move_counter = Counter(session.moves)
        result["summary"] = {
            "total_moves": len(session.moves),
            "unique_moves": len(move_counter),
            "move_distribution": dict(move_counter.most_common()),
        }

    return [TextContent(type="text", text=str(result))]


async def handle_check_solution(args: dict[str, Any]) -> list[TextContent]:
    """
    Verify whether the cube is solved.

    Intent: Provide a binary success criterion. The LLM must determine
    when to stop attempting moves and verify solution.

    Evaluation:
    - is_solved: Primary success metric
    - unsolved_details: Progress toward goal (how many pieces are correct)

    Design: Solved state is defined as each face having uniform color.
    Mismatched position count provides a progress metric when unsolved.

    Semantics: This is a read-only verification. The LLM may call this
    periodically to check progress, but should not rely on it for state
    information (use observe instead).
    """
    session_id = args["session_id"]
    show_details = args.get("show_details", False)

    if session_id not in sessions:
        raise ValueError(f"Session not found: {session_id}")

    session = sessions[session_id]
    is_solved = session.cube.is_done()

    result = {
        "session_id": session_id,
        "is_solved": is_solved,
    }

    if show_details and not is_solved:
        # Get current state and analyze unsolved pieces
        state = session.cube.get()
        expected_solved_state = "YYYYYYYYYRRRRRRRRRGGGGGGGGGOOOOOOOOOBBBBBBBBBWWWWWWWWW"

        # Count mismatched positions
        mismatches = sum(1 for a, b in zip(state, expected_solved_state) if a != b)
        result["unsolved_details"] = {
            "mismatched_positions": mismatches,
            "total_positions": len(state),
            "solved_percentage": (1 - mismatches / len(state)) * 100,
        }

    return [TextContent(type="text", text=str(result))]


async def main():
    """
    MCP server entry point.

    Intent: Establish stdio-based communication with MCP client.
    The server runs as a subprocess, receiving JSON-RPC requests and
    returning tool results.

    Semantics: Uses async I/O for non-blocking request handling.
    Session state persists in the sessions dictionary for the
    duration of the server process.
    """
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
