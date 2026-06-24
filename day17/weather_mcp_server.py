"""
MCP-сервер для погоды.
Использует Open-Meteo API (бесплатный, без ключа).
Реализует MCP протокол с JSON-RPC 2.0.
"""

import json
import uuid
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class MCPTool:
    """Инструмент MCP"""
    name: str
    description: str
    input_schema: Dict[str, Any]


class WeatherMCPServer:
    """
    MCP-сервер для получения погоды через Open-Meteo API.
    Реализует JSON-RPC 2.0 поверх HTTP.
    """

    def __init__(self, host: str = "localhost", port: int = 8003):
        self.host = host
        self.port = port
        self.sessions: Dict[str, Dict] = {}
        self.tools = self._register_tools()

    def _register_tools(self) -> List[MCPTool]:
        """
        Регистрация инструментов MCP.
        """
        return [
            MCPTool(
                name="get_current_weather",
                description="Получить текущую погоду для указанного города",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города на русском или английском (например: Moscow, London, Paris, Москва)"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial"],
                            "description": "Единицы измерения: metric (Celsius, km/h) или imperial (Fahrenheit, mph)",
                            "default": "metric"
                        }
                    },
                    "required": ["city"]
                }
            ),
            MCPTool(
                name="get_weather_forecast",
                description="Получить прогноз погоды на N дней для указанного города",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города на русском или английском"
                        },
                        "days": {
                            "type": "integer",
                            "description": "Количество дней прогноза (1-7)",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 7
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial"],
                            "description": "Единицы измерения",
                            "default": "metric"
                        }
                    },
                    "required": ["city"]
                }
            )
        ]

    def _geocode_city(self, city: str) -> Dict:
        """
        Получение координат города через Open-Meteo Geocoding API.
        """
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": city,
            "count": 1,
            "language": "ru",
            "format": "json"
        }

        print(f"🌐 [GEO] Поиск города: {city}")

        try:
            response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                return {"error": f"Ошибка геокодинга: {response.status_code}"}

            data = response.json()

            if not data.get("results"):
                return {"error": f"Город '{city}' не найден"}

            result = data["results"][0]

            return {
                "name": result.get("name", city),
                "country": result.get("country", ""),
                "latitude": result["latitude"],
                "longitude": result["longitude"]
            }

        except requests.exceptions.Timeout:
            return {"error": "Таймаут при геокодинге"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Ошибка геокодинга: {str(e)}"}
        except Exception as e:
            return {"error": f"Неизвестная ошибка: {str(e)}"}

    def _get_current_weather(self, city: str, units: str = "metric") -> Dict:
        """Получение текущей погоды через wttr.in."""
        url = f"http://wttr.in/{city}"
        params = {"format": "j1", "lang": "ru", "m": ""}

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return {"error": f"Ошибка API: {response.status_code}"}

            data = response.json()

            # Извлекаем текущие данные
            current = data.get("current_condition", [{}])[0]
            location = data.get("nearest_area", [{}])[0]

            # Температура
            temp = current.get("temp_C", "N/A")
            if units == "imperial":
                temp = current.get("temp_F", "N/A")

            # Ветер
            wind = current.get("windspeedKmph", "N/A")
            wind_units = "км/ч"
            if units == "imperial":
                wind = current.get("windspeedMiles", "N/A")
                wind_units = "миль/ч"

            return {
                "city": location.get("areaName", [{}])[0].get("value", city),
                "country": location.get("country", [{}])[0].get("value", ""),
                "temperature": {
                    "value": temp,
                    "units": "°C" if units == "metric" else "°F"
                },
                "condition": current.get("weatherDesc", [{}])[0].get("value", "N/A"),
                "humidity": f"{current.get('humidity', 'N/A')}%",
                "wind": {
                    "speed": wind,
                    "units": wind_units
                },
                "feels_like": current.get("FeelsLikeC", "N/A") if units == "metric" else current.get("FeelsLikeF",
                                                                                                     "N/A"),
                "last_updated": current.get("observation_time", ""),
                "source": "wttr.in"
            }

        except requests.exceptions.Timeout:
            return {"error": "Таймаут при запросе к wttr.in"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Ошибка запроса: {str(e)}"}
        except Exception as e:
            return {"error": f"Неизвестная ошибка: {str(e)}"}

    def _get_current_weather_v1(self, city: str, units: str = "metric") -> Dict:
        """
        Получение текущей погоды через Open-Meteo API.
        """
        print(f"🔍 [TOOL] get_current_weather: city={city}, units={units}")

        # 1. Получаем координаты
        geo_result = self._geocode_city(city)

        if "error" in geo_result:
            return {"error": geo_result["error"]}

        lat = geo_result["latitude"]
        lon = geo_result["longitude"]
        name = geo_result["name"]
        country = geo_result["country"]

        print(f"🌐 [API] Получение погоды для {name} ({lat}, {lon})")

        # 2. Получаем текущую погоду
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "timezone": "auto",
            "temperature_unit": "celsius" if units == "metric" else "fahrenheit",
            "wind_speed_unit": "kmh" if units == "metric" else "mph"
        }

        try:
            response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                return {"error": f"Ошибка получения погоды: {response.status_code}"}

            data = response.json()
            current = data.get("current_weather", {})

            if not current:
                return {"error": "Нет данных о погоде"}

            # Коды погоды Open-Meteo
            weather_codes = {
                0: "☀️ Ясно",
                1: "🌤️ Преимущественно ясно",
                2: "⛅ Переменная облачность",
                3: "☁️ Облачно",
                45: "🌫️ Туман",
                48: "🌫️ Туман с изморозью",
                51: "🌧️ Морось слабая",
                53: "🌧️ Морось умеренная",
                55: "🌧️ Морось сильная",
                56: "🌧️ Морось ледяная слабая",
                57: "🌧️ Морось ледяная сильная",
                61: "🌧️ Дождь слабый",
                63: "🌧️ Дождь умеренный",
                65: "🌧️ Дождь сильный",
                66: "🌧️ Дождь ледяной слабый",
                67: "🌧️ Дождь ледяной сильный",
                71: "🌨️ Снегопад слабый",
                73: "🌨️ Снегопад умеренный",
                75: "🌨️ Снегопад сильный",
                77: "🌨️ Снежные зёрна",
                80: "🌧️ Ливень слабый",
                81: "🌧️ Ливень умеренный",
                82: "🌧️ Ливень сильный",
                85: "🌨️ Снегопад с ливнем слабый",
                86: "🌨️ Снегопад с ливнем сильный",
                95: "⛈️ Гроза слабая",
                96: "⛈️ Гроза с градом слабая",
                99: "⛈️ Гроза с градом сильная"
            }

            weather_code = current.get("weathercode", 0)
            condition = weather_codes.get(weather_code, f"Код: {weather_code}")

            temp = current.get("temperature", "N/A")
            wind = current.get("windspeed", "N/A")

            return {
                "city": name,
                "country": country,
                "temperature": {
                    "value": temp,
                    "units": "°C" if units == "metric" else "°F"
                },
                "condition": condition,
                "wind": {
                    "speed": wind,
                    "units": "км/ч" if units == "metric" else "миль/ч"
                },
                "last_updated": current.get("time", ""),
                "units": units,
                "source": "Open-Meteo"
            }

        except requests.exceptions.Timeout:
            return {"error": "Таймаут при запросе к API погоды"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Ошибка запроса: {str(e)}"}
        except Exception as e:
            return {"error": f"Неизвестная ошибка: {str(e)}"}

    def _get_weather_forecast(self, city: str, days: int = 3, units: str = "metric") -> Dict:
        """Получение прогноза погоды через wttr.in."""
        url = f"http://wttr.in/{city}"
        params = {"format": "j1", "lang": "ru", "m": ""}

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return {"error": f"Ошибка API: {response.status_code}"}

            data = response.json()
            location = data.get("nearest_area", [{}])[0]
            forecasts = data.get("weather", [])[:days]

            result = {
                "city": location.get("areaName", [{}])[0].get("value", city),
                "country": location.get("country", [{}])[0].get("value", ""),
                "forecast": []
            }

            for day in forecasts:
                date = day.get("date", "")
                max_temp = day.get("maxtempC", "N/A") if units == "metric" else day.get("maxtempF", "N/A")
                min_temp = day.get("mintempC", "N/A") if units == "metric" else day.get("mintempF", "N/A")

                # Берём состояние на полдень (индекс 4 из 8)
                hourly = day.get("hourly", [])
                condition = "N/A"
                if len(hourly) > 4:
                    condition = hourly[4].get("weatherDesc", [{}])[0].get("value", "N/A")

                result["forecast"].append({
                    "date": date,
                    "max_temperature": {
                        "value": max_temp,
                        "units": "°C" if units == "metric" else "°F"
                    },
                    "min_temperature": {
                        "value": min_temp,
                        "units": "°C" if units == "metric" else "°F"
                    },
                    "condition": condition
                })

            return result

        except Exception as e:
            return {"error": f"Ошибка прогноза: {str(e)}"}

    def _get_weather_forecast_v1(self, city: str, days: int = 3, units: str = "metric") -> Dict:
        """
        Получение прогноза погоды через Open-Meteo API.
        """
        print(f"🔍 [TOOL] get_weather_forecast: city={city}, days={days}, units={units}")

        # 1. Получаем координаты
        geo_result = self._geocode_city(city)

        if "error" in geo_result:
            return {"error": geo_result["error"]}

        lat = geo_result["latitude"]
        lon = geo_result["longitude"]
        name = geo_result["name"]
        country = geo_result["country"]

        print(f"🌐 [API] Получение прогноза для {name} ({lat}, {lon})")

        # 2. Получаем прогноз
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,weathercode",
            "timezone": "auto",
            "forecast_days": min(days, 7),
            "temperature_unit": "celsius" if units == "metric" else "fahrenheit"
        }

        try:
            response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                return {"error": f"Ошибка получения прогноза: {response.status_code}"}

            data = response.json()
            daily = data.get("daily", {})

            if not daily:
                return {"error": "Нет данных о прогнозе"}

            # Коды погоды Open-Meteo
            weather_codes = {
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

            forecast = []
            times = daily.get("time", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            codes = daily.get("weathercode", [])

            for i in range(len(times)):
                code = codes[i] if i < len(codes) else 0
                forecast.append({
                    "date": times[i] if i < len(times) else "N/A",
                    "max_temperature": {
                        "value": max_temps[i] if i < len(max_temps) else "N/A",
                        "units": "°C" if units == "metric" else "°F"
                    },
                    "min_temperature": {
                        "value": min_temps[i] if i < len(min_temps) else "N/A",
                        "units": "°C" if units == "metric" else "°F"
                    },
                    "condition": weather_codes.get(code, f"Код: {code}")
                })

            return {
                "city": name,
                "country": country,
                "forecast": forecast,
                "units": units,
                "source": "Open-Meteo"
            }

        except requests.exceptions.Timeout:
            return {"error": "Таймаут при запросе к API прогноза"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Ошибка запроса: {str(e)}"}
        except Exception as e:
            return {"error": f"Неизвестная ошибка: {str(e)}"}

    def _handle_tools_list(self, session_id: str, request_id: int) -> Dict:
        """
        Обработка метода tools/list.
        """
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
            "result": {
                "tools": tools_data
            }
        }

    def _handle_tools_call(self, session_id: str, params: Dict, request_id: int) -> Dict:
        """
        Обработка метода tools/call.
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "get_current_weather":
            city = arguments.get("city")
            units = arguments.get("units", "metric")
            result = self._get_current_weather(city, units)

        elif tool_name == "get_weather_forecast":
            city = arguments.get("city")
            days = arguments.get("days", 3)
            units = arguments.get("units", "metric")
            result = self._get_weather_forecast(city, days, units)

        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }

        if "error" in result:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": result["error"]
                }
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
        """
        Обработка метода initialize.
        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": "Weather MCP Server",
                    "version": "1.0.0"
                }
            }
        }

    def handle_request(self, request: Dict, headers: Dict) -> tuple:
        """
        Обработка входящего JSON-RPC запроса.
        Возвращает (response_data, headers).
        """
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        # Логируем входящий запрос
        print(f"\n{'='*60}")
        print(f"📥 [MCP] ВХОДЯЩИЙ ЗАПРОС")
        print(f"{'='*60}")
        print(f"📌 Метод: {method}")
        print(f"📌 ID: {request_id}")
        if params:
            print(f"📌 Параметры: {json.dumps(params, ensure_ascii=False)[:500]}")
        if headers.get("Mcp-Session-Id"):
            print(f"📌 Session: {headers['Mcp-Session-Id'][:20]}...")
        print(f"{'='*60}")

        # Получаем или создаём session_id
        session_id = headers.get("Mcp-Session-Id")
        if not session_id:
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = {"created_at": datetime.now()}

        # Маршрутизация методов
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
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

        # Логируем исходящий ответ
        print(f"\n📤 [MCP] ИСХОДЯЩИЙ ОТВЕТ")
        print(f"{'='*60}")
        response_preview = json.dumps(response, ensure_ascii=False, indent=2)
        if len(response_preview) > 500:
            print(f"{response_preview[:500]}...")
        else:
            print(response_preview)
        print(f"{'='*60}\n")

        # Возвращаем JSON
        response_json = json.dumps(response, ensure_ascii=False)

        headers = {
            "Content-Type": "application/json",
            "Mcp-Session-Id": session_id
        }

        return response_json, headers


# ============================================================
# HTTP-сервер
# ============================================================

import http.server
import socketserver


class MCPHandler(http.server.BaseHTTPRequestHandler):
    """HTTP обработчик для MCP-сервера."""

    server_instance = None

    def do_POST(self):
        """Обработка POST запросов к /mcp."""
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
            request,
            dict(self.headers)
        )

        self.send_response(200)
        for key, value in response_headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(response_data.encode('utf-8'))

    def do_GET(self):
        """Обработка GET запросов."""
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
        """Отключаем лишние логи."""
        pass


def run_server(host="localhost", port=8003):
    """Запуск MCP-сервера."""

    server = WeatherMCPServer(host, port)
    MCPHandler.server_instance = server

    with socketserver.TCPServer((host, port), MCPHandler) as httpd:
        print(f"\n{'='*60}")
        print(f"🌤️  MCP Weather Server (Open-Meteo)")
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