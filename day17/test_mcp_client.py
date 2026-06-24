"""
Тест MCP клиента
"""

import json
from weather_mcp_client import MCPWeatherClient


def test():
    print("=" * 60)
    print("🧪 ТЕСТ MCP КЛИЕНТА")
    print("=" * 60)

    client = MCPWeatherClient()

    # 1. Подключение
    if not client.connect():
        print("❌ Не удалось подключиться")
        return

    # 2. Получение инструментов
    tools = client.get_tools()
    print(f"\n📦 Инструменты: {[t['name'] for t in tools]}")

    # 3. Проверка погоды
    print("\n📌 Проверка: погода в Москве")
    result = client.call_tool("get_current_weather", {"city": "Moscow"})
    print(f"\n📊 Результат:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test()