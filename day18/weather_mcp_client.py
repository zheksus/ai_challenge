"""
Консольный клиент для Weather MCP Server.
Подключается к серверу, получает список тулов и вызывает их по командам из консоли.
"""

import json
import sys
import shlex

import requests


class MCPClient:
    def __init__(self, server_url: str = "http://localhost:8003"):
        self.mcp_url = f"{server_url}/mcp"
        self.session_id = None
        self.tools = []

    def _send(self, method: str, params: dict = None, req_id: int = 1) -> dict:
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": req_id}
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        try:
            r = requests.post(self.mcp_url, json=payload, headers=headers, timeout=30)
            if "Mcp-Session-Id" in r.headers:
                self.session_id = r.headers["Mcp-Session-Id"]
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}"}
            return r.json()
        except requests.exceptions.ConnectionError:
            return {"error": f"Connection refused to {self.mcp_url}"}
        except Exception as e:
            return {"error": str(e)}

    def connect(self) -> bool:
        result = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "weather-cli", "version": "1.0"}
        })
        if "result" in result:
            print(f"Connected: {result['result'].get('serverInfo', {}).get('name', '?')}")
            return True
        print(f"Connect error: {result.get('error', 'unknown')}")
        return False

    def fetch_tools(self) -> list:
        result = self._send("tools/list", {}, 2)
        self.tools = result.get("result", {}).get("tools", [])
        return self.tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        result = self._send("tools/call", {"name": name, "arguments": arguments}, 3)
        if "error" in result:
            return {"error": result["error"]}
        content = result.get("result", {}).get("content", [])
        if content and isinstance(content, list):
            text = content[0].get("text", "")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
        return {"error": "No result"}


def print_result(data: dict, indent: int = 0):
    prefix = " " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}:")
                print_result(v, indent + 2)
            else:
                print(f"{prefix}{k}: {v}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"{prefix}[{i}]:")
            print_result(item, indent + 2)
    else:
        print(f"{prefix}{data}")


def parse_args(text: str, schema: dict) -> dict:
    """Парсит строку вида city=Moscow, interval_seconds=300 в словарь,
    учитывая типы из схемы."""
    props = schema.get("properties", {})
    args = {}
    parts = shlex.split(text.replace(",", " "))
    for part in parts:
        if "=" not in part:
            if "city" in props and "city" not in args:
                args["city"] = part
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        prop = props.get(key, {})
        ptype = prop.get("type", "string")
        if ptype == "integer":
            try:
                val = int(val)
            except ValueError:
                pass
        elif ptype == "number":
            try:
                val = float(val)
            except ValueError:
                pass
        args[key] = val
    return args


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MCP Weather CLI")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8003)
    cli_args = parser.parse_args()

    url = f"http://{cli_args.host}:{cli_args.port}"
    client = MCPClient(url)

    if not client.connect():
        sys.exit(1)

    tools = client.fetch_tools()
    if not tools:
        print("No tools available")
        sys.exit(1)

    print(f"\nAvailable tools ({len(tools)}):")
    for t in tools:
        req = t.get("inputSchema", {}).get("required", [])
        print(f"  {t['name']} ({', '.join(req)}) — {t['description']}")
    print("\nType a tool name + arguments, or 'help', or 'quit'.")
    print("Example: get_current_weather city=Moscow units=metric")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line in ("help", "h"):
            for t in tools:
                req = t.get("inputSchema", {}).get("required", [])
                opt = [k for k in t.get("inputSchema", {}).get("properties", {}) if k not in req]
                print(f"\n{t['name']}")
                print(f"  {t['description']}")
                print(f"  Required: {', '.join(req) if req else 'none'}")
                print(f"  Optional: {', '.join(opt) if opt else 'none'}")
            print()
            continue

        parts = line.split(maxsplit=1)
        name = parts[0]
        args_str = parts[1] if len(parts) > 1 else ""

        tool = next((t for t in tools if t["name"] == name), None)
        if not tool:
            print(f"Unknown tool: {name}. Try: help")
            continue

        schema = tool.get("inputSchema", {})
        arguments = parse_args(args_str, schema)

        # fill defaults
        for k, v in schema.get("properties", {}).items():
            if k not in arguments and "default" in v:
                arguments[k] = v["default"]

        print(f"\n→ {name}({json.dumps(arguments, ensure_ascii=False)})")
        result = client.call_tool(name, arguments)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print_result(result)
        print()


if __name__ == "__main__":
    main()
