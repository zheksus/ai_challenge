"""
mcp_tool_caller.py — вызов инструментов MCP Calculator Server
"""

import requests
import json


class MCPCalculatorClient:
    def __init__(self, server_url: str = "http://localhost:8000", endpoint: str = "/mcp"):
        self.full_url = f"{server_url.rstrip('/')}{endpoint}"
        self.session_id = None
        self.tools = []

    def _make_request(self, method: str, params: dict = None, request_id: int = 1) -> dict:
        """Отправка JSON-RPC запроса к MCP серверу."""
        if params is None:
            params = {}

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }

        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        resp = requests.post(self.full_url, json=payload, headers=headers, timeout=30)

        # Сохраняем session_id из ответа
        if "Mcp-Session-Id" in resp.headers:
            self.session_id = resp.headers["Mcp-Session-Id"]

        # Парсим SSE ответ
        if resp.status_code == 200:
            for line in resp.text.split('\n'):
                if line.startswith('data: '):
                    return json.loads(line[6:])
            return {"error": "No data in SSE response"}
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}

    def initialize(self) -> bool:
        """Инициализация MCP соединения."""
        result = self._make_request(
            "initialize",
            {
                "protocolVersion": "0.1.0",
                "capabilities": {},
                "clientInfo": {"name": "python-client", "version": "1.0"}
            },
            1
        )

        if "result" in result:
            print("✅ MCP инициализирован")
            return True
        print(f"❌ Ошибка: {result}")
        return False

    def get_tools(self) -> list:
        """Получение списка инструментов."""
        if not self.session_id:
            print("⚠️ Сначала вызовите initialize()")
            return []

        result = self._make_request("tools/list", {}, 2)

        if "result" in result and "tools" in result["result"]:
            self.tools = result["result"]["tools"]
            print(f"📦 Получено {len(self.tools)} инструментов")
            return self.tools
        return []

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Вызов инструмента."""
        if not self.session_id:
            print("⚠️ Сначала вызовите initialize()")
            return {"error": "Not initialized"}

        result = self._make_request(
            "tools/call",
            {"name": name, "arguments": arguments},
            3
        )

        if "result" in result:
            return result["result"]
        return {"error": result.get("error", "Unknown error")}


def main():
    print("=" * 60)
    print("🧪 MCP CALCULATOR — ВЫЗОВ ИНСТРУМЕНТОВ")
    print("=" * 60)

    client = MCPCalculatorClient()

    # 1. Инициализация
    if not client.initialize():
        return

    # 2. Получение инструментов
    tools = client.get_tools()
    print(f"\n📦 Инструменты: {[t['name'] for t in tools]}")

    # 3. Вызов инструментов
    print("\n" + "=" * 60)
    print("🔧 ВЫЗОВ ИНСТРУМЕНТОВ")
    print("=" * 60)

    # add
    result = client.call_tool("add", {"a": 5, "b": 3})
    print(f"\n📌 add(5, 3) → {result}")

    # subtract
    result = client.call_tool("subtract", {"a": 10, "b": 4})
    print(f"📌 subtract(10, 4) → {result}")

    # multiply
    result = client.call_tool("multiply", {"a": 4, "b": 7})
    print(f"📌 multiply(4, 7) → {result}")

    # divide
    result = client.call_tool("divide", {"a": 10, "b": 2})
    print(f"📌 divide(10, 2) → {result}")


if __name__ == "__main__":
    main()