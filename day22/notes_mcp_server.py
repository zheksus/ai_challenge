"""
MCP-сервер для заметок.
Хранит заметки в SQLite.
Реализует MCP протокол с JSON-RPC 2.0.
"""

import json
import os
import uuid
import sqlite3
import http.server
import socketserver
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]


class NotesMCPServer:
    def __init__(self, host: str = "localhost", port: int = 8004):
        self.host = host
        self.port = port
        self.sessions: Dict[str, Dict] = {}
        self.tools = self._register_tools()
        self._init_db()

    def _register_tools(self) -> List[MCPTool]:
        return [
            MCPTool(
                name="create_note",
                description="Создать новую заметку с заголовком и содержимым",
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Заголовок заметки"
                        },
                        "content": {
                            "type": "string",
                            "description": "Содержимое заметки"
                        }
                    },
                    "required": ["title", "content"]
                }
            ),
            MCPTool(
                name="list_notes",
                description="Получить список всех заметок (id, заголовок, дата создания)",
                input_schema={
                    "type": "object",
                    "properties": {}
                }
            ),
            MCPTool(
                name="read_note",
                description="Получить полное содержимое заметки по её ID",
                input_schema={
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "ID заметки"
                        }
                    },
                    "required": ["id"]
                }
            )
        ]

    def _init_db(self):
        self._db_conn = sqlite3.connect("notes.db", check_same_thread=False)
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._db_conn.commit()

    def _handle_create_note(self, title: str, content: str) -> Dict:
        now = datetime.now().isoformat()
        cursor = self._db_conn.execute(
            "INSERT INTO notes (title, content, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (title, content, now, now)
        )
        self._db_conn.commit()
        note_id = cursor.lastrowid
        return {
            "id": note_id,
            "title": title,
            "message": f"Заметка '{title}' создана (ID: {note_id})"
        }

    def _handle_list_notes(self) -> Dict:
        rows = self._db_conn.execute(
            "SELECT id, title, created_at FROM notes ORDER BY id DESC"
        ).fetchall()
        notes = []
        for r in rows:
            notes.append({
                "id": r[0],
                "title": r[1],
                "created_at": r[2]
            })
        return {"notes": notes, "total": len(notes)}

    def _handle_read_note(self, note_id: int) -> Dict:
        row = self._db_conn.execute(
            "SELECT id, title, content, created_at, updated_at FROM notes WHERE id = ?",
            (note_id,)
        ).fetchone()
        if not row:
            return {"error": f"Заметка с ID {note_id} не найдена"}
        return {
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "created_at": row[3],
            "updated_at": row[4]
        }

    def _handle_tools_list(self, session_id: str, request_id: int) -> Dict:
        tools_data = []
        for tool in self.tools:
            tools_data.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            })
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools_data}
        }

    def _handle_tools_call(self, session_id: str, params: Dict, request_id: int) -> Dict:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "create_note":
            title = arguments.get("title")
            content = arguments.get("content")
            result = self._handle_create_note(title, content)
        elif tool_name == "list_notes":
            result = self._handle_list_notes()
        elif tool_name == "read_note":
            note_id = arguments.get("id")
            result = self._handle_read_note(note_id)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
            }

        if "error" in result:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": result["error"]}
            }

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False, indent=2)
                    }
                ],
                "structuredContent": result,
                "isError": False
            }
        }

    def _handle_initialize(self, session_id: str, params: Dict, request_id: int) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "Notes MCP Server", "version": "1.0.0"}
            }
        }

    def handle_request(self, request: Dict, headers: Dict) -> tuple:
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        print(f"\n{'='*60}")
        print(f"📥 [NOTES-MCP] ВХОДЯЩИЙ ЗАПРОС")
        print(f"{'='*60}")
        print(f"📌 Метод: {method}")
        print(f"📌 ID: {request_id}")
        if params:
            print(f"📌 Параметры: {json.dumps(params, ensure_ascii=False)[:500]}")
        print(f"{'='*60}")

        session_id = headers.get("Mcp-Session-Id")
        if not session_id:
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = {"created_at": datetime.now()}

        if method == "initialize":
            response = self._handle_initialize(session_id, params, request_id)
        elif method == "tools/list":
            response = self._handle_tools_list(session_id, request_id)
        elif method == "tools/call":
            response = self._handle_tools_call(session_id, params, request_id)
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }

        print(f"\n📤 [NOTES-MCP] ИСХОДЯЩИЙ ОТВЕТ")
        response_preview = json.dumps(response, ensure_ascii=False, indent=2)
        if len(response_preview) > 500:
            print(f"{response_preview[:500]}...")
        else:
            print(response_preview)
        print(f"{'='*60}\n")

        response_json = json.dumps(response, ensure_ascii=False)
        headers = {"Content-Type": "application/json", "Mcp-Session-Id": session_id}
        return response_json, headers


class MCPHandler(http.server.BaseHTTPRequestHandler):
    server_instance = None

    def do_POST(self):
        if self.path != "/mcp":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        response_data, response_headers = self.server_instance.handle_request(
            request, dict(self.headers)
        )

        self.send_response(200)
        for key, value in response_headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(response_data.encode('utf-8'))

    def do_GET(self):
        if self.path == "/mcp":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(b"event: message\ndata: {}\n\n")
            self.wfile.flush()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def run_server(host="localhost", port=8004):
    server = NotesMCPServer(host, port)
    MCPHandler.server_instance = server

    with socketserver.TCPServer((host, port), MCPHandler) as httpd:
        print(f"\n{'='*60}")
        print(f"📝 MCP Notes Server")
        print(f"{'='*60}")
        print(f"✅ Сервер запущен на http://{host}:{port}")
        print(f"📌 MCP эндпоинт: http://{host}:{port}/mcp")
        print(f"\n📦 Доступные инструменты:")
        for tool in server.tools:
            print(f"  - {tool.name}: {tool.description}")
        print(f"\n⚠️  Нажмите Ctrl+C для остановки")
        print(f"{'='*60}\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Остановка сервера...")


if __name__ == "__main__":
    run_server()