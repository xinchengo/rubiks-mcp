#!/usr/bin/env python3
"""
LLM Evaluation Platform for Rubik's Cube Sequential Reasoning

Intent: Evaluate Large Language Models on sequential action and reasoning
capabilities through the Rubik's Cube domain.

Architecture:
    Follows the Observe → Reason → Act → Verify loop:
    1. Observe: Get current cube state
    2. Reason: LLM analyzes state and decides on moves
    3. Act: Apply the suggested move sequence
    4. Verify: Check if solved, repeat if not

Evaluation Dimensions:
    - Success rate: Can the LLM solve the cube?
    - Move efficiency: How many moves compared to scramble?
    - Loop detection: Does the LLM get stuck in repetitive patterns?
    - Reasoning quality: Does the LLM show coherent reasoning?
    - Time efficiency: How long does it take to solve?

Usage:
    python demo_langchain_agent.py
"""

import asyncio
import os
import re
import subprocess
import threading
import json
from typing import Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from zhipuai import ZhipuAI

# Load environment variables
load_dotenv()

# Model configuration
MODEL_NAME = "glm-4.5-air"


class MCPServerClient:
    """
    Client for connecting to an MCP stdio server.

    Intent: Provide a lightweight interface to the Rubik's Cube MCP server
    without requiring the full MCP client library.

    Design: Manages subprocess communication, sends JSON-RPC requests,
    parses responses, and maintains session state.
    """

    def __init__(self, server_path: str):
        self.server_path = server_path
        self.proc = None
        self.tools = []
        self._initialized = False
        self._stderr_drainer_thread = None
        self._stderr_output = []

    async def start(self):
        """Start the MCP server subprocess."""
        self.proc = subprocess.Popen(
            [self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Start stderr drain thread to prevent blocking
        self._stderr_drainer_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_drainer_thread.start()

        # Initialize connection
        await self._send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "glm-evaluation", "version": "1.0.0"},
            },
        })

        # Get available tools
        response = await self._send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })
        self.tools = response.get("result", {}).get("tools", [])
        self._initialized = True

    def _drain_stderr(self):
        """Drain stderr in a background thread to prevent subprocess blocking."""
        while self.proc and self.proc.poll() is None:
            try:
                line = self.proc.stderr.readline()
                if not line:
                    break
                self._stderr_output.append(line.strip())
            except Exception:
                break

    async def _send_request(self, request: dict) -> dict:
        """Send a JSON-RPC request and return the response."""
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("Server process not running")

        # Write request
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()

        # Read response using asyncio to avoid blocking
        loop = asyncio.get_event_loop()
        response_line = await loop.run_in_executor(None, self.proc.stdout.readline)
        if not response_line:
            # Print any stderr output for debugging
            if self._stderr_output:
                print(f"Server stderr output: {''.join(self._stderr_output)}")
            raise RuntimeError("No response from server")
        return json.loads(response_line)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        """Call a tool by name with arguments."""
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        response = await self._send_request({
            "jsonrpc": "2.0",
            "id": hash(f"{name}_{asyncio.get_event_loop().time()}"),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        })

        content = response.get("result", {}).get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            # Parse the text content (it's a dict as string)
            text = content[0].get("text", "{}")
            return eval(text) if text else {}
        return {}

    async def close(self):
        """Close the server subprocess."""
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait()


class LLMAgent:
    """
    LLM-based agent for solving Rubik's Cube.

    Intent: Wrapper around the ZhipuAI GLM model to handle cube reasoning.

    Design: Maintains conversation history, parses LLM responses for moves,
    and provides structured prompts for the observe-act cycle.
    """

    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
        # Track moves suggested by LLM (separate from moves applied to cube)
        self.suggested_moves = []

    def _extract_moves(self, response_text: str) -> Optional[str]:
        """
        Extract move sequence from LLM response.

        Intent: Parse the LLM's natural language output to find the
        actual moves it wants to apply.

        Design: Look for common patterns like "moves:", "apply:", or
        standalone move sequences (e.g., "R U R'").
        """
        response_text = response_text.strip()

        # Pattern 0: Use regex to extract move sequence directly (most reliable)
        move_pattern = r'(?<!\S)([RLUDFB][\'2]?(?:\s+[RLUDFB][\'2]?)+)(?!\S)'
        matches = re.findall(move_pattern, response_text, re.IGNORECASE)
        if matches:
            return matches[0].upper()

        # Pattern 1: Look for explicit "moves:" or "apply:" declarations
        for prefix in ["moves:", "apply:", "apply moves:", "move sequence:", "MOVES:"]:
            if prefix.lower() in response_text.lower():
                idx = response_text.lower().find(prefix.lower())
                after_prefix = response_text[idx + len(prefix):].strip()
                # Take first line or until next sentence
                moves_line = after_prefix.split("\n")[0].split(".")[0].strip()
                # Validate that it looks like moves
                if self._is_valid_move_sequence(moves_line):
                    return moves_line

        # Pattern 2: Look for standalone lines with move notation
        for line in response_text.split("\n"):
            line = line.strip()
            # Skip lines that are too long or clearly not moves
            if self._is_valid_move_sequence(line):
                return line

        return None

    def _is_valid_move_sequence(self, text: str) -> bool:
        """Check if a text string looks like a valid move sequence."""
        tokens = text.strip().split()
        if not tokens:
            return False

        # Check if all tokens look like moves
        move_pattern = re.compile(r'^[RLUDFB][\'2]?$', re.IGNORECASE)
        valid_tokens = [move_pattern.match(token) for token in tokens]

        # At least 50% should be valid moves
        valid_ratio = sum(1 for v in valid_tokens if v) / len(tokens)
        return valid_ratio >= 0.5

    async def get_move_sequence(
        self,
        current_state: dict,
        move_history: list[str],
        attempt: int
    ) -> tuple[Optional[str], str]:
        """
        Get the next move sequence from the LLM.

        Intent: Ask the LLM to reason about the current state and decide
        what moves to apply.

        Args:
            current_state: Current cube state (faces, is_solved, etc.)
            move_history: List of moves already applied
            attempt: Which attempt number this is

        Returns:
            (moves_sequence, full_response): The parsed moves and full LLM response
        """
        faces = current_state.get('faces', {})
        is_solved = current_state.get('is_solved', False)
        move_count = current_state.get('move_count', 0)

        if is_solved:
            return None, "Cube is already solved!"

        # Build the prompt
        # Build prompt with a more natural cube representation
        # Group face colors for cleaner display
        def _summarize_face(face_list: list[str]) -> str:
            """Summarize face by dominant color."""
            from collections import Counter
            counts = Counter(face_list)
            dominant = counts.most_common(1)[0][0] if counts else '?'
            mixed = len(set(face_list)) > 1
            return f"{dominant}{' (mixed)' if mixed else ''}"

        u_face = _summarize_face(faces.get('U', []))
        f_face = _summarize_face(faces.get('F', []))
        r_face = _summarize_face(faces.get('R', []))
        b_face = _summarize_face(faces.get('B', []))
        l_face = _summarize_face(faces.get('L', []))
        d_face = _summarize_face(faces.get('D', []))

        grid = current_state.get("grid", "")

        prompt = f"""Rubik's Cube (3D visualization):

{grid}
Status: {'SOLVED' if is_solved else 'UNSOLVED'} | Moves applied: {move_count}
Your previous suggestions: {' '.join(self.suggested_moves[-10:]) if self.suggested_moves else 'None'}
Moves actually applied: {' '.join(move_history[-10:]) if move_history else 'None'}

Task: Provide 1-3 moves to solve. End with: MOVES: <moves>
Valid moves: R L U D F B (add ' for prime, 2 for double)"""

        # Add to conversation history (keep last 4 exchanges)
        self.conversation_history.append({"role": "user", "content": prompt})
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]

        # Call the model with timeout
        loop = asyncio.get_event_loop()
        try:
            # Use closure to properly capture client, model, and history
            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    stream=False,
                )
            response = await asyncio.wait_for(
                loop.run_in_executor(None, _make_api_call),
                timeout=30.0  # Reduced for glm-4-flash
            )
            response_text = response.choices[0].message.content
        except asyncio.TimeoutError:
            print("[LLM] Timeout - using empty move sequence")
            return None, "Timeout waiting for LLM response"
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return None, f"Error: {e}"

        # Add to history
        self.conversation_history.append({"role": "assistant", "content": response_text})

        # Extract moves
        moves = self._extract_moves(response_text)

        # Track moves suggested by LLM (separate from applied moves)
        if moves:
            # Parse individual moves and add to suggested history
            suggested = moves.split()
            self.suggested_moves.extend(suggested)
            # Keep last 20 suggestions
            if len(self.suggested_moves) > 20:
                self.suggested_moves = self.suggested_moves[-20:]

        # Debug: print if extraction included prefix
        if moves and ('MOVES:' in moves or 'moves:' in moves):
            print(f"[Reason] WARNING: Prefix still in moves: {moves}")

        return moves, response_text


async def run_evaluation(
    scramble: str = "R U",
    max_attempts: int = 10,
    verbose: bool = True
) -> dict:
    """
    Run an evaluation session for the LLM agent.

    Intent: Execute a complete evaluation cycle following the
    Observe → Reason → Act → Verify loop.

    Args:
        scramble: The scramble to apply (e.g., "R U")
        max_attempts: Maximum number of observe-act cycles
        verbose: Print detailed progress

    Returns:
        Dictionary with evaluation results and metrics
    """
    if verbose:
        print("=" * 60)
        print(f"Rubik's Cube LLM Evaluation Platform")
        print(f"Model: {MODEL_NAME}")
        print(f"Scramble: {scramble}")
        print(f"Max attempts: {max_attempts}")
        print("=" * 60)

    # Load API key
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise ValueError("GLM_API_KEY not found in environment variables")

    # Initialize agent
    agent = LLMAgent(api_key=api_key, model=MODEL_NAME)

    # Start MCP server (use relative path)
    server_path = str(Path(__file__).parent / "rubiks_mcp" / "server.py")
    mcp_client = MCPServerClient(server_path)
    await mcp_client.start()

    if verbose:
        print(f"\n[Server] Connected with {len(mcp_client.tools)} tools")

    # Create session
    if verbose:
        print(f"\n[Session] Creating with scramble: {scramble}")

    session_result = await mcp_client.call_tool("create_session", {
        "size": 3,
        "scramble": scramble,
        "observation": "full",
    })

    session_id = session_result["session_id"]
    initial_state = session_result['state']

    if verbose:
        print(f"[Session] ID: {session_id[:8]}...")
        print(f"[Session] Initial state: {initial_state}")

    # Initial observation with grid format
    observe_result = await mcp_client.call_tool("observe", {
        "session_id": session_id,
        "format": "grid",
    })

    current_state = observe_result
    move_history = []
    llm_responses = []

    # Main evaluation loop: Observe → Reason → Act → Verify
    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"\n{'='*20} Attempt {attempt}/{max_attempts} {'='*20}")

        # Check if already solved
        if current_state.get('is_solved', False):
            if verbose:
                print("[Result] CUBE SOLVED!")
            break

        # Step 1: Observe (already have state)
        if verbose:
            print("[Observe] Current state retrieved")

        # Step 2: Reason - Ask LLM what moves to apply
        if verbose:
            print("[Reason] Asking LLM for next moves...")

        moves, llm_response = await agent.get_move_sequence(
            current_state,
            move_history,
            attempt
        )

        llm_responses.append({
            "attempt": attempt,
            "response": llm_response,
            "moves": moves
        })

        if verbose:
            print(f"[Reason] LLM suggested moves: {moves if moves else 'None'}")

        if not moves:
            if verbose:
                print("[Reason] LLM didn't provide valid moves, ending evaluation")
            break

        # Step 3: Act - Apply the moves
        if verbose:
            print(f"[Act] Applying moves: {moves}")

        rotate_result = await mcp_client.call_tool("rotate", {
            "session_id": session_id,
            "moves": moves,
        })

        applied_moves = rotate_result.get('moves_applied', [])
        move_history.extend(applied_moves)

        if verbose:
            print(f"[Act] Applied {len(applied_moves)} move(s)")
            print(f"[Act] Total moves so far: {len(move_history)}")

        # Step 4: Verify - Observe again and check if solved
        if verbose:
            print("[Verify] Observing new state...")

        observe_result = await mcp_client.call_tool("observe", {
            "session_id": session_id,
            "format": "grid",
        })

        current_state = observe_result

        if current_state.get('is_solved', False):
            if verbose:
                print("[Verify] CUBE SOLVED!")
            break

        if verbose:
            # Check solution progress
            check_result = await mcp_client.call_tool("check_solution", {
                "session_id": session_id,
                "show_details": True,
            })
            if "unsolved_details" in check_result:
                details = check_result["unsolved_details"]
                print(f"[Verify] Progress: {details['solved_percentage']:.1f}% solved")

    # Get final metrics
    metrics_result = await mcp_client.call_tool("get_metrics", {
        "session_id": session_id,
        "include_loop_analysis": True,
    })

    history_result = await mcp_client.call_tool("get_history", {
        "session_id": session_id,
        "format": "string",
    })

    # Compile results
    results = {
        "success": current_state.get('is_solved', False),
        "attempts": attempt,
        "total_moves": metrics_result.get('total_moves', 0),
        "unique_moves": metrics_result.get('unique_moves', 0),
        "move_distribution": metrics_result.get('move_distribution', {}),
        "scramble": scramble,
        "scramble_moves": metrics_result.get('scramble_moves', 0),
        "move_history": move_history,
        "move_history_string": history_result.get('moves_string', ''),
        "observation_count": metrics_result.get('observation_count', 0),
        "loop_analysis": metrics_result.get('loop_analysis', {}),
        "llm_responses": llm_responses,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Success: {'YES' if results['success'] else 'NO'}")
        print(f"Scramble: {scramble} ({metrics_result.get('scramble_moves', 0)} moves)")
        print(f"Total moves: {results['total_moves']}")
        print(f"Unique moves: {results['unique_moves']}")
        print(f"Attempts: {results['attempts']}")
        print(f"Observations: {results['observation_count']}")
        print(f"Has loops: {results['loop_analysis'].get('has_loops', False)}")
        print(f"Move history: {results['move_history_string']}")
        print("=" * 60)

    # Cleanup
    await mcp_client.close()

    return results


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM sequential reasoning on Rubik's Cube"
    )
    parser.add_argument(
        "--scramble",
        type=str,
        default="R U",
        help="Scramble sequence to apply (default: 'R U')"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Maximum number of observe-act cycles (default: 10)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    args = parser.parse_args()

    results = await run_evaluation(
        scramble=args.scramble,
        max_attempts=args.max_attempts,
        verbose=not args.quiet
    )

    # Exit with appropriate code
    return 0 if results['success'] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    if exit_code is not None:
        exit(exit_code)
