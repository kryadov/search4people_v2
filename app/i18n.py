"""Bilingual UI strings (English / Russian) for chat-facing copy.

Code, comments, and developer-facing messages stay in English. Only strings shown
to the end user inside the chat interface live here.
"""

from __future__ import annotations

from typing import Literal

Locale = Literal["en", "ru"]
DEFAULT_LOCALE: Locale = "en"

_TRANSLATIONS: dict[str, dict[Locale, str]] = {
    "disclaimer": {
        "en": (
            "**Disclaimer.** This assistant aggregates publicly available information only. "
            "Do not use it to harass, dox, or otherwise harm individuals. You are responsible "
            "for complying with the laws of your jurisdiction (GDPR, CCPA, etc.) and the terms "
            "of service of every platform involved."
        ),
        "ru": (
            "**Предупреждение.** Ассистент собирает только публично доступную информацию. "
            "Не используйте его для преследования, доксинга или иного причинения вреда. Вы "
            "несёте ответственность за соблюдение законов вашей юрисдикции (GDPR и др.) и "
            "условий использования каждой задействованной площадки."
        ),
    },
    "ask_who": {
        "en": "Who are we looking for? Please share the **first name** and **last name**.",
        "ru": "Кого ищем? Укажите **имя** и **фамилию**.",
    },
    "ask_more_details_prefix": {
        "en": "I found multiple candidates. To narrow it down, could you tell me",
        "ru": "Я нашёл несколько кандидатов. Чтобы сузить поиск, подскажите,",
    },
    "candidates_heading": {
        "en": "**Candidates found ({count}):**",
        "ru": "**Найденные кандидаты ({count}):**",
    },
    "options_heading": {
        "en": "**Options seen in candidates** (for *{attribute}*):",
        "ru": "**Варианты из сниппетов кандидатов** (для *{attribute}*):",
    },
    "narrowing_reply_hint": {
        "en": (
            "Reply with **#N** to pick that candidate, paste one of the options above, "
            "or type any other value."
        ),
        "ru": (
            "Ответьте **#N**, чтобы выбрать конкретного кандидата, скопируйте один из "
            "вариантов выше или введите своё значение."
        ),
    },
    "ask_disambiguate_prefix": {
        "en": "Quick check to disambiguate:",
        "ru": "Уточняющий вопрос для устранения неоднозначности:",
    },
    "searching_platform": {
        "en": "Searching {platform}…",
        "ru": "Ищу на {platform}…",
    },
    "fetching_url": {
        "en": "Fetching {url}",
        "ru": "Извлекаю {url}",
    },
    "expanding_search": {
        "en": "No candidates yet — expanding the search to all configured platforms.",
        "ru": "Кандидатов пока нет — расширяю поиск на все настроенные площадки.",
    },
    "building_profile": {
        "en": "Building the profile…",
        "ru": "Собираю профиль…",
    },
    "confirm_profile": {
        "en": (
            "Here's the profile I assembled. Does it look right? Reply **yes** to save, "
            "**no** to keep searching, or describe what should change."
        ),
        "ru": (
            "Вот собранный профиль. Всё верно? Ответьте **да**, чтобы сохранить, "
            "**нет**, чтобы продолжить поиск, или опишите, что поправить."
        ),
    },
    "not_found": {
        "en": "I could not find a confident match within the configured iteration limit.",
        "ru": "Не удалось найти уверенное совпадение в пределах настроенного лимита итераций.",
    },
    "profile_saved": {
        "en": "Profile saved.",
        "ru": "Профиль сохранён.",
    },
    "guard_blocked": {
        "en": (
            "I can't help with this request. It appears to involve harmful or "
            "prohibited use (e.g. harassment, doxing, or targeting a minor). "
            "This tool is for legitimate, lawful people-search only."
        ),
        "ru": (
            "Не могу помочь с этим запросом. Похоже, он связан с недопустимым "
            "использованием (преследование, доксинг или поиск несовершеннолетних). "
            "Инструмент предназначен только для законного поиска людей."
        ),
    },
    "language_set": {
        "en": "Language set to English.",
        "ru": "Язык переключён на русский.",
    },
    "language_toggle_hint": {
        "en": "Type `/ru` for Russian or `/en` for English at any time.",
        "ru": "Введите `/ru` для русского или `/en` для английского в любой момент.",
    },
    "tags_line_label": {
        "en": "Tags:",
        "ru": "Теги:",
    },
}


def t(key: str, locale: Locale = DEFAULT_LOCALE, **kwargs: object) -> str:
    """Translate `key` into `locale`, formatting with `kwargs`."""
    bundle = _TRANSLATIONS.get(key)
    if bundle is None:
        return key
    template = bundle.get(locale) or bundle.get(DEFAULT_LOCALE) or key
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError):
            return template
    return template


def detect_locale_command(message: str) -> Locale | None:
    """Return a locale if the message is a language-switch command, else None."""
    stripped = message.strip().lower()
    if stripped in {"/ru", "/russian", "/русский"}:
        return "ru"
    if stripped in {"/en", "/english", "/английский"}:
        return "en"
    return None
