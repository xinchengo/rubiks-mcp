#!/usr/bin/env python3
"""Test script for the Rubik's Cube MCP server evaluation tools."""

import asyncio
import subprocess
import json


async def test_evaluation_tools():
    """Test the MCP server evaluation tools."""
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
    print(f"Server: {init_response.get('result', {}).get('serverInfo', 'N/A')}")

    # List tools
    print("\n=== Available Tools ===")
    tools_response = send_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
    })
    tools = tools_response.get("result", {}).get("tools", [])
    for tool in tools:
        print(f"  - {tool['name']}")

    # Create a session
    print("\n=== Create Session ===")
    create_response = send_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "create_session",
            "arguments": {
                "size": 3,
                "scramble": "R U R' U' R' F R2",
                "observation": "full",
            },
        },
    })
    create_result = eval(create_response["result"]["content"][0]["text"])
    session_id = create_result["session_id"]
    print(f"Session ID: {session_id}")
    print(f"Scramble applied: {create_result['scramble_applied']}")

    # Check solution (should be false)
    print("\n=== Check Solution ===")
    check_response = send_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "check_solution",
            "arguments": {
                "session_id": session_id,
                "show_details": True,
            },
        },
    })
    check_result = eval(check_response["result"]["content"][0]["text"])
    print(f"Is solved: {check_result['is_solved']}")
    if "unsolved_details" in check_result:
        details = check_result["unsolved_details"]
        print(f"Mismatched positions: {details['mismatched_positions']}/{details['total_positions']}")
        print(f"Solved percentage: {details['solved_percentage']:.1f}%")

    # Make some moves (intentionally create a loop)
    print("\n=== Making Moves ===")
    move_sequence = "R U R' U' R U R' U' R U R' U'"  # Repeated pattern
    rotate_response = send_request({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "rotate",
            "arguments": {
                "session_id": session_id,
                "moves": move_sequence,
            },
        },
    })
    rotate_result = eval(rotate_response["result"]["content"][0]["text"])
    print(f"Moves applied: {rotate_result['moves_applied']}")

    # Get history
    print("\n=== Get History ===")
    history_response = send_request({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "get_history",
            "arguments": {
                "session_id": session_id,
                "format": "string",
            },
        },
    })
    history_result = eval(history_response["result"]["content"][0]["text"])
    print(f"History: {history_result['moves_string']}")
    print(f"Count: {history_result['count']}")

    # Get metrics
    print("\n=== Get Metrics ===")
    metrics_response = send_request({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "get_metrics",
            "arguments": {
                "session_id": session_id,
                "include_loop_analysis": True,
            },
        },
    })
    metrics_result = eval(metrics_response["result"]["content"][0]["text"])
    print(f"Total moves: {metrics_result['total_moves']}")
    print(f"Unique moves: {metrics_result['unique_moves']}")
    print(f"Observation count: {metrics_result['observation_count']}")
    print(f"Elapsed time: {metrics_result['time_tracking']['elapsed_time_display']}")
    print(f"Average time per move: {metrics_result['time_tracking']['average_time_per_move']}")
    print(f"Has loops: {metrics_result['loop_analysis']['has_loops']}")
    print(f"Loops detected: {metrics_result['loop_analysis']['loops_detected']}")

    if metrics_result['loop_analysis']['loop_patterns']:
        print("\nLoop patterns:")
        for i, pattern in enumerate(metrics_result['loop_analysis']['loop_patterns'][:3], 1):
            print(f"  {i}. '{pattern['pattern']}' - {pattern['occurrences']} occurrences")

    # Get history summary
    print("\n=== Get History Summary ===")
    summary_response = send_request({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "get_history",
            "arguments": {
                "session_id": session_id,
                "format": "summary",
            },
        },
    })
    summary_result = eval(summary_response["result"]["content"][0]["text"])
    print(f"Total moves: {summary_result['summary']['total_moves']}")
    print(f"Unique moves: {summary_result['summary']['unique_moves']}")
    print(f"Move distribution: {summary_result['summary']['move_distribution']}")

    # Clean up
    proc.stdin.close()
    proc.terminate()
    proc.wait()
    print("\n=== All evaluation tests passed! ===")


if __name__ == "__main__":
    asyncio.run(test_evaluation_tools())
