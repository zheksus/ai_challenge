import json

import requests
import sseclient


class MCPWeatherClient:
    """
    Клиент для MCP Weather Server.
    Подключается к нашему MCP-серверу и предоставляет инструменты для агента.
    """

    def __init__(self, server_url: str = "http://localhost:8003"):
        self.base_url = server_url
        self.mcp_url = f"{server_url}/mcp"
        self.session_id = None
        self.tools = []
        self.connected = False

    def _call_openmeteo(self, city: str) -> dict:
        """
        Альтернативный API для погоды (Open-Meteo).
        Бесплатный, без ключа, работает стабильно.
        """
        # 1. Получаем координаты города через геокодинг
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "count": 1,
            "language": "ru",
            "format": "json"
        }

        print(f"🌐 [API] Поиск города: {city}")

        try:
            geo_response = requests.get(geo_url, params=geo_params, timeout=15)
            if geo_response.status_code != 200:
                return {"error": f"Город '{city}' не найден"}

            geo_data = geo_response.json()
            if not geo_data.get("results"):
                return {"error": f"Город '{city}' не найден"}

            result = geo_data["results"][0]
            lat = result["latitude"]
            lon = result["longitude"]
            name = result["name"]
            country = result.get("country", "")

            print(f"🌐 [API] Найден: {name} ({lat}, {lon})")

            # 2. Получаем погоду
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": "true",
                "timezone": "auto",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh"
            }

            print(f"🌐 [API] Получение погоды для {name}")
            weather_response = requests.get(weather_url, params=weather_params, timeout=15)

            if weather_response.status_code != 200:
                return {"error": "Не удалось получить погоду"}

            weather_data = weather_response.json()
            current = weather_data.get("current_weather", {})

            # Определяем состояние погоды по коду
            weather_code = current.get("weathercode", 0)
            weather_conditions = {
                0: "☀️ Ясно",
                1: "🌤️ Преимущественно ясно",
                2: "⛅ Переменная облачность",
                3: "☁️ Облачно",
                45: "🌫️ Туман",
                51: "🌧️ Морось",
                61: "🌧️ Дождь",
                71: "🌨️ Снегопад",
                80: "🌧️ Ливень",
                95: "⛈️ Гроза"
            }
            condition = weather_conditions.get(weather_code, f"Код: {weather_code}")

            return {
                "city": name,
                "country": country,
                "temperature": {
                    "value": current.get("temperature", "N/A"),
                    "units": "°C"
                },
                "wind": {
                    "speed": current.get("windspeed", "N/A"),
                    "units": "км/ч"
                },
                "condition": condition,
                "last_updated": current.get("time", ""),
                "source": "Open-Meteo"
            }

        except requests.exceptions.Timeout:
            return {"error": "Таймаут при запросе к API погоды"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Ошибка запроса: {str(e)}"}
        except Exception as e:
            return {"error": f"Неизвестная ошибка: {str(e)}"}

    def _send_request(self, method: str, params: dict = None, request_id: int = 1) -> dict:
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
            "Accept": "application/json"  # 👈 ТОЛЬКО JSON
        }

        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        print(f"\n📤 [CLIENT] → {method} (id={request_id})")

        try:
            response = requests.post(
                self.mcp_url,
                json=payload,
                headers=headers,
                timeout=30
            )

            if "Mcp-Session-Id" in response.headers:
                self.session_id = response.headers["Mcp-Session-Id"]
                print(f"📤 [CLIENT] Session ID: {self.session_id[:20]}...")

            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}"}

            # 👇 ПРОСТО ПАРСИМ JSON
            return response.json()

        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.ConnectionError as e:
            return {"error": f"Connection error: {e}"}
        except Exception as e:
            return {"error": str(e)}

    def _send_request_sse(self, method: str, params: dict = None, request_id: int = 1) -> dict:
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

        print(f"\n📤 [CLIENT] → {method} (id={request_id})")

        try:
            # 👇 НЕ ИСПОЛЬЗУЕМ STREAM
            response = requests.post(
                self.mcp_url,
                json=payload,
                headers=headers,
                timeout=30
            )

            if "Mcp-Session-Id" in response.headers:
                self.session_id = response.headers["Mcp-Session-Id"]
                print(f"📤 [CLIENT] Session ID: {self.session_id[:20]}...")

            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}"}

            response_text = response.text
            print(f"📤 [CLIENT] Получено {len(response_text)} байт")

            # Парсим SSE
            for line in response_text.split('\n'):
                if line.startswith('data: '):
                    data_str = line[6:].strip()
                    if data_str:
                        try:
                            return json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

            # Пробуем как JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass

            return {"error": f"No valid response"}

        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.ConnectionError as e:
            return {"error": f"Connection error: {e}"}
        except Exception as e:
            return {"error": str(e)}


    def _send_request_v3_sse(self, method: str, params: dict = None, request_id: int = 1) -> dict:
        """Отправка JSON-RPC запроса к MCP серверу с использованием SSE клиента."""
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

        print(f"\n📤 [CLIENT] → {method} (id={request_id})")

        try:
            response = requests.post(
                self.mcp_url,
                json=payload,
                headers=headers,
                timeout=60,
                stream=True
            )

            if "Mcp-Session-Id" in response.headers:
                self.session_id = response.headers["Mcp-Session-Id"]

            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}

            # 👇 ИСПОЛЬЗУЕМ SSE КЛИЕНТ
            client = sseclient.SSEClient(response)
            for event in client.events():
                if event.data:
                    try:
                        return json.loads(event.data)
                    except json.JSONDecodeError:
                        continue
                # Если получили пустое событие — пробуем дальше

            return {"error": "No SSE events received"}

        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.ConnectionError as e:
            return {"error": f"Connection error: {e}"}
        except Exception as e:
            return {"error": str(e)}

    def _send_request_v2_line_by_line(self, method: str, params: dict = None, request_id: int = 1) -> dict:
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

        print(f"\n📤 [CLIENT] → {method} (id={request_id})")

        try:
            response = requests.post(
                self.mcp_url,
                json=payload,
                headers=headers,
                timeout=60,
                stream=True
            )

            print(f"📤 [CLIENT] Статус: {response.status_code}")
            print(f"📤 [CLIENT] Content-Type: {response.headers.get('content-type', 'unknown')}")

            if "Mcp-Session-Id" in response.headers:
                self.session_id = response.headers["Mcp-Session-Id"]
                print(f"📤 [CLIENT] Session ID: {self.session_id[:20]}...")

            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}

            # 👇 ЧИТАЕМ ВЕСЬ ОТВЕТ
            response_text = ""
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    response_text += chunk

            print(f"📤 [CLIENT] Получено {len(response_text)} байт")
            print(f"📤 [CLIENT] Ответ (первые 500 символов):\n{response_text[:500]}")
            print(f"📤 [CLIENT] Ответ (последние 200 символов):\n{response_text[-200:]}")

            # 👇 ПАРСИМ SSE
            # Ищем все data: строки
            data_lines = []
            for line in response_text.split('\n'):
                if line.startswith('data: '):
                    data_str = line[6:].strip()
                    if data_str and data_str != "{}":
                        data_lines.append(data_str)
                        print(f"📤 [CLIENT] Найдена data: {data_str[:100]}...")

            if data_lines:
                # Берём последнюю data (обычно это ответ)
                last_data = data_lines[-1]
                try:
                    return json.loads(last_data)
                except json.JSONDecodeError as e:
                    print(f"⚠️ [CLIENT] Ошибка парсинга JSON: {e}")
                    return {"error": f"Invalid JSON: {last_data[:100]}"}

            # Если data не найдены, пробуем парсить весь текст как JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass

            return {"error": f"No data found in response: {response_text[:100]}"}

        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.ConnectionError as e:
            return {"error": f"Connection error: {e}"}
        except Exception as e:
            return {"error": str(e)}


    def _send_request_v1(self, method: str, params: dict = None, request_id: int = 1) -> dict:
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

        print(f"\n📤 [CLIENT] → {method} (id={request_id})")

        try:
            response = requests.post(
                self.mcp_url,
                json=payload,
                headers=headers,
                timeout=30
            )

            if "Mcp-Session-Id" in response.headers:
                self.session_id = response.headers["Mcp-Session-Id"]

            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}

            response_text = response.text.strip()

            # Пробуем разные форматы
            # 1. SSE формат
            if response_text.startswith('data: '):
                data_str = response_text[6:].strip()
                if data_str:
                    try:
                        return json.loads(data_str)
                    except json.JSONDecodeError:
                        pass

            # 2. SSE с несколькими строками
            for line in response_text.split('\n'):
                if line.startswith('data: '):
                    data_str = line[6:].strip()
                    if data_str and data_str != "{}":
                        try:
                            return json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

            # 3. Обычный JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass

            return {"error": f"No valid response: {response_text[:100]}"}

        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.ConnectionError as e:
            return {"error": f"Connection error: {e}"}
        except Exception as e:
            return {"error": str(e)}

    def connect(self) -> bool:
        """Подключение к MCP серверу."""
        print("\n📌 Подключение к MCP Weather Server...")

        result = self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent-client", "version": "1.0"}
            },
            1
        )

        if "error" in result:
            print(f"❌ Ошибка подключения: {result['error']}")
            return False

        if "result" in result:
            self.connected = True
            server_name = result["result"].get("serverInfo", {}).get("name", "Unknown")
            print(f"✅ MCP Weather Server подключён ({server_name})")
            return True

        print(f"⚠️ Неожиданный ответ: {result}")
        return False

    def get_tools(self) -> list:
        """Получение списка инструментов."""
        if not self.connected:
            print("⚠️ Сначала вызовите connect()")
            return []

        result = self._send_request("tools/list", {}, 2)

        if "result" in result and "tools" in result["result"]:
            self.tools = result["result"]["tools"]
            print(f"📦 Получено {len(self.tools)} инструментов")
            return self.tools

        return []

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Вызов MCP инструмента."""
        if not self.connected:
            return {"error": "MCP not connected"}

        print(f"\n🔧 [CLIENT] Вызов инструмента: {name}")
        print(f"🔧 [CLIENT] Аргументы: {json.dumps(arguments, ensure_ascii=False)}")

        result = self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
            3
        )

        if "error" in result:
            return {"error": result["error"]}

        if "result" in result:
            content = result["result"].get("content", [])
            if content and isinstance(content, list):
                text = content[0].get("text", "")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw": text, "text": text}

        return {"error": "No result"}

    def get_tools_prompt(self) -> str:
        """Формирование промпта с описанием инструментов для LLM."""
        if not self.tools:
            return ""

        lines = [
            "### ДОСТУПНЫЕ MCP ИНСТРУМЕНТЫ (ПОГОДА):",
            "Ты можешь использовать следующие инструменты для получения информации о погоде:"
        ]

        for tool in self.tools:
            name = tool.get("name")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            lines.append(f"\n🔧 **{name}**: {description}")
            if properties:
                lines.append("   Параметры:")
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    is_required = "✅" if param_name in required else "➖"
                    param_desc = param_info.get("description", "")
                    lines.append(f"     {is_required} `{param_name}` ({param_type}): {param_desc}")

        return "\n".join(lines)