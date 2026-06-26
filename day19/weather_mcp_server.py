"""
MCP-сервер для погоды.
Использует Open-Meteo API (бесплатный, без ключа).
Реализует MCP протокол с JSON-RPC 2.0.
"""

import json
import os
import uuid
import sqlite3
import threading
import time
import requests
from datetime import datetime, timedelta
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
        self._init_db()
        self._monitors: Dict[str, dict] = {}
        self._monitor_threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}

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
            ),
            MCPTool(
                name="start_weather_monitoring",
                description="Запустить фоновый сбор погоды для города по расписанию. Данные сохраняются в БД.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        },
                        "interval_seconds": {
                            "type": "integer",
                            "description": "Интервал сбора в секундах",
                            "default": 60,
                            "minimum": 1
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
            ),
            MCPTool(
                name="stop_weather_monitoring",
                description="Остановить фоновый сбор погоды для указанного города",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        }
                    },
                    "required": ["city"]
                }
            ),
            MCPTool(
                name="get_weather_history",
                description="Получить историю собранных данных о погоде для города",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города (если не указать — вернуть всё)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Максимум записей",
                            "default": 100,
                            "minimum": 1,
                            "maximum": 1000
                        }
                    }
                }
            ),
            MCPTool(
                name="get_weather_recommendation",
                description="Дать рекомендацию по одежде на основе погоды в городе (куртка, зонт)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
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
            ),
            MCPTool(
                name="save_weather_report",
                description="Сохранить отчёт о погоде в файл (включает текущую погоду, прогноз и рекомендацию)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Имя файла для сохранения (будет создан в рабочей папке сервера)"
                        },
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        },
                        "days": {
                            "type": "integer",
                            "description": "Количество дней прогноза",
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
                    "required": ["filename", "city"]
                }
            )
        ]

    def _init_db(self):
        self._db_conn = sqlite3.connect("weather_monitor.db", check_same_thread=False)
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS weather_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                temperature REAL,
                condition TEXT,
                humidity TEXT,
                wind_speed REAL,
                wind_units TEXT,
                units TEXT,
                raw_data TEXT
            )
        """)
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS weather_monitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                monitor_key TEXT UNIQUE NOT NULL,
                city TEXT NOT NULL,
                interval_seconds INTEGER NOT NULL,
                units TEXT DEFAULT 'metric',
                active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        self._db_conn.commit()

    def _save_snapshot(self, city: str, weather: Dict, units: str):
        temp = None
        cond = None
        hum = None
        ws = None
        wu = None
        try:
            temp = float(weather.get("temperature", {}).get("value", 0))
        except (TypeError, ValueError):
            pass
        cond = weather.get("condition")
        hum = weather.get("humidity")
        try:
            ws = float(weather.get("wind", {}).get("speed", 0))
        except (TypeError, ValueError):
            pass
        wu = weather.get("wind", {}).get("units")
        self._db_conn.execute(
            """INSERT INTO weather_snapshots
               (city, timestamp, temperature, condition, humidity, wind_speed, wind_units, units, raw_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (city, datetime.now().isoformat(), temp, cond, hum, ws, wu, units,
             json.dumps(weather, ensure_ascii=False)),
        )
        self._db_conn.commit()

    def _monitor_loop(self, monitor_key: str, city: str, interval: int, units: str):
        stop_event = self._stop_events.get(monitor_key)
        while stop_event and not stop_event.is_set():
            weather = self._get_current_weather(city, units)
            if "error" not in weather:
                self._save_snapshot(city, weather, units)
                print(f"📸 [MONITOR] Snapshot saved for {city} at {datetime.now().isoformat()}")
            else:
                print(f"⚠️ [MONITOR] Failed to fetch weather for {city}: {weather['error']}")
            stop_event.wait(interval)

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

    def _handle_start_monitoring(self, city: str, interval_seconds: int, units: str) -> Dict:
        key = city.lower().strip()
        if key in self._monitors:
            return {"message": f"Мониторинг для {city} уже запущен (интервал: {self._monitors[key]['interval']} сек)"}

        stop_event = threading.Event()
        self._stop_events[key] = stop_event
        self._monitors[key] = {"city": city, "interval": interval_seconds, "units": units}
        t = threading.Thread(target=self._monitor_loop, args=(key, city, interval_seconds, units), daemon=True)
        self._monitor_threads[key] = t
        t.start()
        return {
            "message": f"Мониторинг погоды для {city} запущен",
            "city": city,
            "interval_seconds": interval_seconds,
            "units": units,
            "status": "active"
        }

    def _handle_stop_monitoring(self, city: str) -> Dict:
        key = city.lower().strip()
        if key not in self._monitors:
            return {"error": f"Мониторинг для {city} не найден"}
        if key in self._stop_events:
            self._stop_events[key].set()
        if key in self._monitor_threads:
            self._monitor_threads[key].join(timeout=5)
        self._monitors.pop(key, None)
        self._stop_events.pop(key, None)
        self._monitor_threads.pop(key, None)
        return {"message": f"Мониторинг для {city} остановлен", "city": city}

    def _handle_get_history(self, city: Optional[str], limit: int) -> Dict:
        if city:
            rows = self._db_conn.execute(
                "SELECT city, timestamp, temperature, condition, humidity, wind_speed, wind_units "
                "FROM weather_snapshots WHERE city = ? ORDER BY id DESC LIMIT ?",
                (city, limit)
            ).fetchall()
        else:
            rows = self._db_conn.execute(
                "SELECT city, timestamp, temperature, condition, humidity, wind_speed, wind_units "
                "FROM weather_snapshots ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        snapshots = []
        for r in rows:
            snapshots.append({
                "city": r[0],
                "timestamp": r[1],
                "temperature": r[2],
                "condition": r[3],
                "humidity": r[4],
                "wind_speed": r[5],
                "wind_units": r[6],
            })
        return {"snapshots": snapshots, "total": len(snapshots)}

    def _get_weather_recommendation(self, city: str, units: str = "metric") -> Dict:
        """Рекомендация по одежде на основе текущей погоды."""
        weather = self._get_current_weather(city, units)
        if "error" in weather:
            return weather

        temp = weather.get("temperature", {}).get("value", 0)
        condition = weather.get("condition", "").lower()
        wind = weather.get("wind", {}).get("speed", "N/A")

        try:
            temp = float(temp)
        except (TypeError, ValueError):
            temp = 0

        recommendation = {
            "city": weather.get("city", city),
            "temperature": temp,
            "condition": weather.get("condition", ""),
            "need_jacket": False,
            "need_umbrella": False,
            "details": [],
            "summary": ""
        }

        # Температура — рекомендация по куртке
        if temp < 10:
            recommendation["need_jacket"] = True
            recommendation["details"].append(f"Холодно ({temp}°C), нужна куртка")
        elif temp < 20:
            recommendation["details"].append(f"Прохладно ({temp}°C), можно лёгкую куртку")
        else:
            recommendation["details"].append(f"Тепло ({temp}°C), куртка не нужна")

        # Проверка на дождь
        rain_keywords = ["дождь", "rain", "ливень", "морось", "drizzle", "гроза", "thunderstorm"]
        if any(kw in condition for kw in rain_keywords):
            recommendation["need_umbrella"] = True
            recommendation["details"].append("Сейчас идёт дождь — возьмите зонт")

        # Ветер
        try:
            wind_val = float(wind)
            if wind_val > 30:
                recommendation["details"].append(f"Сильный ветер ({wind_val} ед.) — ветровка не помешает")
        except (TypeError, ValueError):
            pass

        # Формируем краткую сводку
        parts = []
        if recommendation["need_jacket"]:
            parts.append("нужна куртка 🧥")
        elif temp < 20:
            parts.append("лёгкая куртка пригодится")
        else:
            parts.append("куртка не нужна")

        if recommendation["need_umbrella"]:
            parts.append("нужен зонт ☂️")

        recommendation["summary"] = f"{recommendation['city']}: {', '.join(parts)}" if parts else \
            f"{recommendation['city']}: погода комфортная, ничего особенного не нужно"

        return recommendation

    def _save_weather_report(self, filename: str, city: str, days: int = 3, units: str = "metric") -> Dict:
        """Форматированный отчёт о погоде в файл."""
        # Защита от path traversal
        safe_name = os.path.basename(filename)
        if not safe_name:
            return {"error": "Некорректное имя файла"}
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), safe_name)

        # Собираем данные
        weather = self._get_current_weather(city, units)
        if "error" in weather:
            return {"error": f"Не удалось получить погоду: {weather['error']}"}

        forecast = self._get_weather_forecast(city, days, units)
        recommendation = self._get_weather_recommendation(city, units)

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = []
        lines.append("=" * 55)
        lines.append(f"  ОТЧЁТ О ПОГОДЕ — {weather.get('city', city).upper()}")
        lines.append(f"  Сформирован: {now}")
        lines.append("=" * 55)
        lines.append("")

        lines.append("--- ТЕКУЩАЯ ПОГОДА ---")
        lines.append(f"  Температура: {weather.get('temperature', {}).get('value', 'N/A')}"
                     f"{weather.get('temperature', {}).get('units', '')}")
        lines.append(f"  Состояние: {weather.get('condition', 'N/A')}")
        lines.append(f"  Влажность: {weather.get('humidity', 'N/A')}")
        wind = weather.get('wind', {})
        lines.append(f"  Ветер: {wind.get('speed', 'N/A')} {wind.get('units', '')}")
        lines.append("")

        if "forecast" in forecast:
            lines.append("--- ПРОГНОЗ ---")
            for day in forecast["forecast"]:
                max_t = day.get("max_temperature", {})
                min_t = day.get("min_temperature", {})
                lines.append(f"  {day.get('date', 'N/A')}: "
                             f"{min_t.get('value', 'N/A')}…{max_t.get('value', 'N/A')}"
                             f"{max_t.get('units', '')}, {day.get('condition', 'N/A')}")
            lines.append("")

        lines.append("--- РЕКОМЕНДАЦИЯ ---")
        rec = recommendation
        for detail in rec.get("details", []):
            lines.append(f"  • {detail}")
        lines.append("")
        lines.append(f"  Итого: {rec.get('summary', 'N/A')}")
        lines.append("")

        lines.append("=" * 55)
        lines.append("  Отчёт сгенерирован Weather MCP Server")
        lines.append("=" * 55)

        content = "\n".join(lines)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return {
                "message": f"Отчёт сохранён: {safe_name}",
                "path": filepath,
                "size_chars": len(content),
                "preview": "\n".join(content.split("\n")[:5])
            }
        except Exception as e:
            return {"error": f"Ошибка записи файла: {str(e)}"}

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

        elif tool_name == "start_weather_monitoring":
            city = arguments.get("city")
            interval = arguments.get("interval_seconds", 60)
            units = arguments.get("units", "metric")
            result = self._handle_start_monitoring(city, interval, units)

        elif tool_name == "stop_weather_monitoring":
            city = arguments.get("city")
            result = self._handle_stop_monitoring(city)

        elif tool_name == "get_weather_history":
            city = arguments.get("city")
            limit = arguments.get("limit", 100)
            result = self._handle_get_history(city, limit)

        elif tool_name == "get_weather_recommendation":
            city = arguments.get("city")
            units = arguments.get("units", "metric")
            result = self._get_weather_recommendation(city, units)

        elif tool_name == "save_weather_report":
            filename = arguments.get("filename")
            city = arguments.get("city")
            days = arguments.get("days", 3)
            units = arguments.get("units", "metric")
            result = self._save_weather_report(filename, city, days, units)

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