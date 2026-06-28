# ============================================================================
# ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ
# ============================================================================
import os
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class UserProfile:
    """Профиль пользователя с предпочтениями"""
    # Базовые данные
    name: str = ""
    profession: str = ""
    company: str = ""

    # Предпочтения стиля общения
    communication_style: str = "дружелюбный"  # официальный, деловой, дружелюбный, неформальный
    preferred_language: str = "russian"       # russian, english, bilingual
    response_format: str = "структурированный" # краткий, подробный, структурированный, с примерами
    tone: str = "нейтральный"                  # нейтральный, профессиональный, эмпатичный, энергичный

    # Ограничения
    avoid: List[str] = field(default_factory=lambda: ["жаргон", "длинные абзацы"])
    detail_level: str = "medium"               # low, medium, high
    max_response_length: int = 1000

    # Дополнительные настройки
    use_emojis: bool = True
    default_temperature: float = 0.7
    timezone: str = "Europe/Moscow"

    # Флаги
    is_initialized: bool = False

    def to_markdown(self) -> str:
        """Экспорт профиля в Markdown формат"""
        lines = [
            "# Профиль пользователя",
            "",
            "## Базовые данные",
            f"- Имя: {self.name or 'Не указано'}",
            f"- Профессия: {self.profession or 'Не указано'}",
            f"- Компания: {self.company or 'Не указано'}",
            "",
            "## Стиль общения",
            f"- Стиль: {self.communication_style}",
            f"- Язык: {self.preferred_language}",
            f"- Тон: {self.tone}",
            f"- Формат ответов: {self.response_format}",
            f"- Использовать эмодзи: {'Да' if self.use_emojis else 'Нет'}",
            "",
            "## Ограничения",
            f"- Уровень детализации: {self.detail_level}",
            f"- Максимальная длина ответа: {self.max_response_length} символов",
            f"- Избегать: {', '.join(self.avoid) if self.avoid else 'Не указано'}",
            "",
            "## Дополнительно",
            f"- Часовой пояс: {self.timezone}",
            f"- Температура по умолчанию: {self.default_temperature}",
            f"- Статус: {'Заполнен' if self.is_initialized else 'Не заполнен'}"
        ]
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, content: str) -> 'UserProfile':
        """Загрузка профиля из Markdown"""
        profile = cls()

        # Поиск значений в тексте
        patterns = {
            "name": r"Имя:\s*(.+?)(?:\n|$)",
            "profession": r"Профессия:\s*(.+?)(?:\n|$)",
            "company": r"Компания:\s*(.+?)(?:\n|$)",
            "communication_style": r"Стиль:\s*(.+?)(?:\n|$)",
            "preferred_language": r"Язык:\s*(.+?)(?:\n|$)",
            "tone": r"Тон:\s*(.+?)(?:\n|$)",
            "response_format": r"Формат ответов:\s*(.+?)(?:\n|$)",
            "detail_level": r"Уровень детализации:\s*(.+?)(?:\n|$)",
            "max_response_length": r"Максимальная длина ответа:\s*(\d+)",
            "use_emojis": r"Использовать эмодзи:\s*(Да|Нет)",
            "timezone": r"Часовой пояс:\s*(.+?)(?:\n|$)",
            "default_temperature": r"Температура по умолчанию:\s*([\d.]+)",
            "avoid": r"Избегать:\s*(.+?)(?:\n|$)"
        }

        for attr, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if attr == "use_emojis":
                    setattr(profile, attr, value == "Да")
                elif attr == "max_response_length":
                    try:
                        setattr(profile, attr, int(value))
                    except:
                        pass
                elif attr == "default_temperature":
                    try:
                        setattr(profile, attr, float(value))
                    except:
                        pass
                elif attr == "avoid":
                    setattr(profile, attr, [v.strip() for v in value.split(',') if v.strip()])
                else:
                    setattr(profile, attr, value)

        profile.is_initialized = True
        return profile

    def to_system_prompt(self) -> str:
        """Формирование системного промпта из профиля"""
        if not self.is_initialized:
            return ""

        lines = [
            "### ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ:",
            f"👤 Имя: {self.name or 'Не указано'}",
            f"💼 Профессия: {self.profession or 'Не указано'}",
            f"🏢 Компания: {self.company or 'Не указано'}",
            "",
            "### ПРЕДПОЧТЕНИЯ ОБЩЕНИЯ:",
            f"🎨 Стиль общения: {self.communication_style}",
            f"📝 Формат ответов: {self.response_format}",
            f"🎭 Тон общения: {self.tone}",
            f"📊 Уровень детализации: {self.detail_level}",
            f"🔤 Язык ответов: {self.preferred_language}",
            f"📏 Максимальная длина: {self.max_response_length} символов",
            f"😊 Использовать эмодзи: {'Да' if self.use_emojis else 'Нет'}",
            "",
            "### ОГРАНИЧЕНИЯ:",
            f"🚫 Избегать: {', '.join(self.avoid) if self.avoid else 'Нет ограничений'}",
            "",
            "### ИНСТРУКЦИЯ:",
            "Отвечай пользователю с учётом его профиля и предпочтений, указанных выше.",
            "Адаптируй стиль, формат и тон ответа в соответствии с профилем."
        ]
        return "\n".join(lines)


class ProfileManager:
    """Управление профилем пользователя"""

    PROFILE_FILE = "PROFILE.md"

    def __init__(self, agent):
        self.agent = agent
        self.profile = self._load_or_create()

    def _load_or_create(self) -> UserProfile:
        """Загрузка профиля из файла или создание нового с опросником"""
        if os.path.exists(self.PROFILE_FILE):
            try:
                with open(self.PROFILE_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                profile = UserProfile.from_markdown(content)
                print(f"📂 Загружен профиль пользователя: {profile.name or 'Без имени'}")
                return profile
            except Exception as e:
                print(f"⚠️ Ошибка загрузки профиля: {e}")

        print("\n" + "=" * 60)
        print("📝 ДОБРО ПОЖАЛОВАТЬ! Давайте создадим ваш профиль.")
        print("=" * 60)
        print("Это поможет мне адаптировать ответы под ваш стиль общения.\n")

        profile = self._run_onboarding()
        self._save_profile(profile)
        return profile

    def _run_onboarding(self) -> UserProfile:
        """Запуск опросника для создания профиля"""
        profile = UserProfile()

        print("📌 Ответьте на несколько вопросов (можно пропустить, нажав Enter):\n")

        # Базовые данные
        profile.name = input("👤 Ваше имя: ").strip()
        profile.profession = input("💼 Ваша профессия: ").strip()
        profile.company = input("🏢 Компания (или 'нет'): ").strip()
        if profile.company.lower() == "нет":
            profile.company = ""

        print("\n🎨 Настройки общения:")

        # Стиль общения
        style_options = ["официальный", "деловой", "дружелюбный", "неформальный"]
        print(f"   Стиль общения: {', '.join(style_options)}")
        style = input("   Выберите (по умолчанию дружелюбный): ").strip().lower()
        if style in style_options:
            profile.communication_style = style

        # Формат ответов
        format_options = ["краткий", "структурированный", "подробный", "с примерами"]
        print(f"   Формат ответов: {', '.join(format_options)}")
        fmt = input("   Выберите (по умолчанию структурированный): ").strip().lower()
        if fmt in format_options:
            profile.response_format = fmt

        # Тон
        tone_options = ["нейтральный", "профессиональный", "эмпатичный", "энергичный"]
        print(f"   Тон общения: {', '.join(tone_options)}")
        tone = input("   Выберите (по умолчанию нейтральный): ").strip().lower()
        if tone in tone_options:
            profile.tone = tone

        # Язык
        lang = input("🌐 Язык ответов (russian/english/bilingual, по умолчанию russian): ").strip().lower()
        if lang in ["russian", "english", "bilingual"]:
            profile.preferred_language = lang

        # Эмодзи
        emoji = input("😊 Использовать эмодзи? (да/нет, по умолчанию да): ").strip().lower()
        if emoji in ["нет", "no"]:
            profile.use_emojis = False

        print("\n⚙️ Ограничения:")

        # Уровень детализации
        detail_options = ["low", "medium", "high"]
        print(f"   Уровень детализации: {', '.join(detail_options)}")
        detail = input("   Выберите (по умолчанию medium): ").strip().lower()
        if detail in detail_options:
            profile.detail_level = detail

        # Чего избегать
        avoid = input("🚫 Что стоит избегать в ответах (через запятую, например: жаргон, длинные абзацы): ").strip()
        if avoid:
            profile.avoid = [v.strip() for v in avoid.split(',') if v.strip()]

        # Длина ответа
        try:
            max_len = input("📏 Максимальная длина ответа (символов, по умолчанию 1000): ").strip()
            if max_len:
                profile.max_response_length = int(max_len)
        except:
            pass

        profile.is_initialized = True

        print("\n✅ Профиль создан!")
        print("   Вы всегда можете изменить его в файле PROFILE.md")

        return profile

    def _save_profile(self, profile: UserProfile):
        """Сохранение профиля в файл"""
        try:
            with open(self.PROFILE_FILE, 'w', encoding='utf-8') as f:
                f.write(profile.to_markdown())
            print(f"📁 Профиль сохранён в {self.PROFILE_FILE}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения профиля: {e}")

    def get_profile_prompt(self) -> str:
        """Получение системного промпта с профилем"""
        return self.profile.to_system_prompt()

    def update_profile(self, **kwargs):
        """Обновление профиля"""
        for key, value in kwargs.items():
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)
        self.profile.is_initialized = True
        self._save_profile(self.profile)

    def show_profile(self) -> str:
        """Показать текущий профиль"""
        return self.profile.to_markdown()
