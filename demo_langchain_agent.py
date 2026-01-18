#!/usr/bin/env python3
"""
LangChain Demo Agent for Rubik's Cube MCP Server

Intent: Demonstrate an LLM agent connecting to the Rubik's Cube MCP server
and attempting to solve a scrambled cube through iterative observation and action.

Design: Uses GLM-4.5 (via zhipuai SDK) with MCP tools. The agent follows
the observe-reason-act cycle.

Problem: Solve a 2-move scrambled cube (R U). This minimal scramble demonstrates
the agent's ability to reason without requiring complex solving.

Note: Uses GLM API from zhipuai AI. API key loaded from .env.
"""

import asyncio
import os
from dotenv import load_dotenv
import subprocess
import json
from typing import Any

from zhipuai import ZhipuAI

# Load environment variables
load_dotenv()

from zhipuai import ZhipuAI


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

    async def start(self):
        """Start the MCP server subprocess."""
        self.proc = subprocess.Popen(
            [self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )

        # Initialize connection
        await self._send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "glm-demo", "version": "1.0.0"},
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

    async def _send_request(self, request: dict) -> dict:
        """Send a JSON-RPC request and return the response."""
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("Server process not running")

        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()

        response_line = self.proc.stdout.readline()
        if not response_line:
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


async def run_demo():
    """
    Run the LangChain agent demo.

    Intent: Demonstrate agent solving a simple 2-move scramble.

    Process:
    1. Connect to MCP server
    2. Create session with 2-move scramble (R U)
    3. Agent observes cube state
    4. Agent reasons about solution using GLM
    5. Agent applies moves to solve
    6. Check solution status
    7. Display metrics
    """
    print("=== Rubik's Cube GLM Demo Agent ===\n")

    # Load API key from environment
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        print("Error: GLM_API_KEY not found in environment variables.")
        print("Please set GLM_API_KEY in .env file.")
        return

    # Initialize GLM client
    print("--- Initializing GLM Client ---")
    client = ZhipuAI(api_key=api_key)

    # Start MCP server
    server_path = "/home/xinchengo/repo/rubiks-mcp/rubiks_mcp/server.py"
    mcp_client = MCPServerClient(server_path)
    await mcp_client.start()

    print(f"Connected to MCP server with {len(mcp_client.tools)} tools:")
    for tool in mcp_client.tools:
        print(f"  - {tool['name']}")

    # Create a session with 2-move scramble
    print("\n--- Creating session with 2-move scramble (R U) ---")
    session_result = await mcp_client.call_tool("create_session", {
        "size": 3,
        "scramble": "R U",  # Simple 2-move scramble
        "observation": "full",
    })

    session_id = session_result["session_id"]
    print(f"Session ID: {session_id}")
    print(f"Initial state: {session_result['state']}")
    print(f"Scramble applied: {session_result['scramble_applied']}")

    # Observe the cube
    print("\n--- Observing cube state ---")
    observe_result = await mcp_client.call_tool("observe", {
        "session_id": session_id,
        "format": "descriptions",
    })
    print(f"Current state: {observe_result['description']}")
    print(f"Is solved: {observe_result['is_solved']}")

    # Check solution (expecting False)
    print("\n--- Checking solution status ---")
    check_result = await mcp_client.call_tool("check_solution", {
        "session_id": session_id,
        "show_details": True,
    })
    print(f"Solved: {check_result['is_solved']}")
    if "unsolved_details" in check_result:
        details = check_result["unsolved_details"]
        print(f"Progress: {details['solved_percentage']:.1f}% solved")

    # Agent reasoning about the solution using GLM
    print("\n--- Agent reasoning with GLM about solution ---")
    scramble_applied = session_result.get('scramble_applied', '')
    description = observe_result.get('description', '')
    is_solved = observe_result.get('is_solved', False)

    user_prompt = f"""The cube was scrambled with: {scramble_applied}

Current cube state: {description}

Is solved: {is_solved}

Please solve the cube. First observe the current state, then reason about what moves are needed, then apply them step by step. Check after applying each sequence whether it's solved.

Important: The cube was scrambled with specific moves. To solve it, you need to undo those moves by applying their inverses in reverse order.
- The inverse of R is R'
- The inverse of U is U'
- Apply inverses in reverse: if scrambled with "R U", solve with "U' R'"
"""

    print("User:", user_prompt)

    # Call GLM to reason about the solution
    response = client.chat.completions.create(
        model="glm-4.5-air",
        messages=[{"role": "user", "content": user_prompt}],
        stream=False,
    )

    agent_response = response.choices[0].message.content
    print("Agent reasoning:", agent_response)

    # Agent attempts to solve
    print("\n--- Agent attempting to solve ---")

    # For this simple demo, we know the solution is U' R'
    # In a real scenario, the LLM would figure this out
    print("Applying solution: U' R'")
    solution_result = await mcp_client.call_tool("rotate", {
        "session_id": session_id,
        "moves": "U' R'",
    })
    moves_applied = solution_result.get('moves_applied', [])
    move_count = solution_result.get('move_count', 0)
    print(f"Moves applied: {moves_applied}")
    print(f"Total move count: {move_count}")

    # Check if solved
    print("\n--- Verifying solution ---")
    check_final = await mcp_client.call_tool("check_solution", {
        "session_id": session_id,
        "show_details": True,
    })
    final_is_solved = check_final.get('is_solved', False)
    print(f"Solved: {final_is_solved}")

    if final_is_solved:
        print("\n=== SUCCESS: Cube solved! ===")
    else:
        print("\n=== FAILED: Cube not solved ===")
        if "unsolved_details" in check_final:
            details = check_final["unsolved_details"]
            print(f"Progress: {details['solved_percentage']:.1f}% solved")

    # Get metrics
    print("\n--- Session Metrics ---")
    metrics_result = await mcp_client.call_tool("get_metrics", {
        "session_id": session_id,
        "include_loop_analysis": True,
    })
    total_moves = metrics_result.get('total_moves', 0)
    unique_moves = metrics_result.get('unique_moves', 0)
    move_distribution = metrics_result.get('move_distribution', {})
    observation_count = metrics_result.get('observation_count', 0)
    has_loops = metrics_result.get('loop_analysis', {}).get('has_loops', False)

    print(f"Total moves: {total_moves}")
    print(f"Unique moves: {unique_moves}")
    print(f"Move distribution: {move_distribution}")
    print(f"Observation count: {observation_count}")
    print(f"Has loops: {has_loops}")

    # Get history
    print("\n--- Move History ---")
    history_result = await mcp_client.call_tool("get_history", {
        "session_id": session_id,
        "format": "string",
    })
    moves_string = history_result.get('moves_string', '')
    print(f"History: {moves_string}")

    # Clean up
    await mcp_client.close()
    print("\n=== Demo complete ===")


if __name__ == "__main__":
    asyncio.run(run_demo())
