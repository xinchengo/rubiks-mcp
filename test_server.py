#!/usr/bin/env python3
"""Simple test script for the Rubik's Cube MCP server."""

import asyncio
import subprocess
import json


async def test_server():
    """Test the MCP server with basic operations."""
    # Start the server
    proc = subprocess.Popen(
        ["/home/xinchengo/repo/rubiks-mcp/rubiks_mcp/server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
    )

    # Helper to send request and get response
    def send_request(request: dict) -> dict:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        response_line = proc.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from server")
        return json.loads(response_line)

    # Initialize
    print("Initializing...")
    init_response = send_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    })
    print(f"Initialize response: {init_response.get('result', {}).get('serverInfo', 'N/A')}")

    # List tools
    print("\nListing tools...")
    tools_response = send_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
    })
    tools = tools_response.get("result", {}).get("tools", [])
    print(f"Available tools: {[t['name'] for t in tools]}")

    # Create a session
    print("\nCreating session...")
    create_response = send_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "create_session",
            "arguments": {
                "size": 3,
                "scramble": "R U R' U'",
                "observation": "full",
            },
        },
    })
    create_result = eval(create_response["result"]["content"][0]["text"])
    session_id = create_result["session_id"]
    print(f"Session ID: {session_id}")
    print(f"State: {create_result['state']}")
    print(f"Is solved: {create_result['is_solved']}")

    # Observe the cube
    print("\nObserving cube (state format)...")
    observe_response = send_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "observe",
            "arguments": {
                "session_id": session_id,
                "format": "state",
            },
        },
    })
    observe_result = eval(observe_response["result"]["content"][0]["text"])
    print(f"State: {observe_result['state']}")
    print(f"Move count: {observe_result['move_count']}")

    # Observe with descriptions
    print("\nObserving cube (descriptions format)...")
    observe_desc_response = send_request({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "observe",
            "arguments": {
                "session_id": session_id,
                "format": "descriptions",
            },
        },
    })
    observe_desc_result = eval(observe_desc_response["result"]["content"][0]["text"])
    print(f"Description: {observe_desc_result['description']}")

    # Rotate the cube
    print("\nRotating cube with R U' R'...")
    rotate_response = send_request({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "rotate",
            "arguments": {
                "session_id": session_id,
                "moves": "R U' R'",
            },
        },
    })
    rotate_result = eval(rotate_response["result"]["content"][0]["text"])
    print(f"Moves applied: {rotate_result['moves_applied']}")
    print(f"Total move count: {rotate_result['move_count']}")
    print(f"Is solved: {rotate_result['is_solved']}")

    # Clean up
    proc.stdin.close()
    proc.terminate()
    proc.wait()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_server())
