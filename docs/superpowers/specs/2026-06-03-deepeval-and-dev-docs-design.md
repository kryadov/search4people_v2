# DeepEval-оценки качества LLM + DEV.md

**Дата:** 2026-06-03
**Статус:** утверждён к реализации

## Цель

1. Добавить набор тестов на базе [DeepEval](https://github.com/confident-ai/deepeval), оценивающих
   качество выводов LLM-узлов графа (а не только детерминированную маршрутизацию,
   которую уже покрывают существующие pytest-тесты).
2. Написать `DEV.md` с двумя разделами:
   - как устроена и как запускается DeepEval-оценка;
   - как менять граф LangGraph при необходимости.

DeepEval-тесты **дополняют**, а не заменяют существующий быстрый набор в `tests/`.

## Что оцениваем

Граф (`app/graph/build.py`) собирает узлы из `app/graph/nodes.py`. LLM реально
вызывают три места:

- `extract_profile_from_page` (`app/tools/extract.py`) — извлекает `PersonProfile`
  из markdown одной страницы; системный промпт прямо запрещает выдумывать факты.
- `build_profile` (`app/graph/nodes.py`) — сливает частичные профили в один
  через `with_structured_output(PersonProfile)`.
- `narrow_query` (`app/graph/nodes.py`) — LLM возвращает JSON-план уточняющего
  вопроса (атрибут, вопрос на нужном locale, варианты).

Покрываем четыре аспекта:

| # | Аспект | Узел | Метрики DeepEval |
|---|--------|------|------------------|
| 1 | Faithfulness / отсутствие галлюцинаций | `extract_profile_from_page` | `FaithfulnessMetric` + `HallucinationMetric` (retrieval_context = markdown страницы) + GEval «не выдумывает факты». Для страницы НЕ про целевого человека — детерминированная проверка: пустой профиль, `confidence="low"`. |
| 2 | Корректность structured output | `build_profile` | GEval «каждое поле подкреплено evidence; evidence непустой; confidence соответствует объёму фактов» + pydantic-валидация результата. |
| 3 | Качество уточняющего вопроса | `narrow_query` → `plan_narrowing` | GEval «вопрос ясен, на нужном locale, спрашивает реально различающий кандидатов атрибут». |
| 4 | Релевантность кандидатов (end-to-end) | весь граф | Живой прогон с реальным search+fetch по известному человеку; программно проходим `interrupt`-ы; GEval «итоговый профиль про нужного человека». Best-effort, толерантный порог. |

## Архитектура и раскладка файлов

```
tests/
  evals/
    __init__.py
    conftest.py                  # eval-фикстуры: судья, skip-if-no-key
    judge.py                     # LangChainJudge(DeepEvalBaseLLM) — обёртка над app/llm.py
    data/
      pages/
        jane_doe_github.md       # сохранённый markdown реальной/синтетической страницы
        jane_doe_linkedin.md
        not_jane_doe.md          # страница НЕ про целевого человека (negative case)
      candidates/
        jane_doe.json            # фиксированный список кандидатов (для narrow/build)
      goldens.json               # ожидаемые факты + список «чего быть НЕ должно»
    test_extract_faithfulness.py
    test_build_profile_quality.py
    test_narrow_query_quality.py
    test_e2e_relevance.py        # живой smoke (двойной opt-in: eval + live)
```

Существующие `tests/conftest.py`, `tests/test_*.py` не трогаем (кроме регистрации
маркеров в `pyproject.toml`).

### Судья (`tests/evals/judge.py`)

```python
class LangChainJudge(DeepEvalBaseLLM):
    """Судья DeepEval, ходящий через тот же провайдер, что и приложение."""
    # load_model()      -> build_chat_model() из app/llm.py
    # get_model_name()  -> settings.llm_model
    # generate(prompt, schema=None)   — sync; при schema использует with_structured_output
    # a_generate(prompt, schema=None) — async-вариант
```

- Судья использует `Settings` приложения (тот же провайдер/ключи), отдельных
  ключей не требует. По умолчанию провайдер — локальная **Ollama** с моделью
  `gpt-oss` (см. `.env.example`), поэтому оценки гоняются локально и бесплатно,
  без каких-либо API-ключей.
- В eval-conftest выставляем `DEEPEVAL_TELEMETRY_OPT_OUT=1` и работаем полностью
  локально — без логина в Confident AI и без сетевой телеметрии.

### Голден-данные

- `data/pages/*.md` — заранее сохранённые markdown-рендеры страниц. Это вход для
  `extract_profile_from_page`; они же служат `retrieval_context` для метрик
  faithfulness/hallucination.
- `data/candidates/jane_doe.json` — фиксированный список кандидатов (формат —
  сериализованные `Candidate`-дикты, как в state) для `narrow_query`/`build_profile`.
- `data/goldens.json` — на каждый кейс: `expected_facts` (что профиль должен
  содержать) и `forbidden_facts` (что модель НЕ должна выдумать).

## Рефакторинг под тестируемость

`narrow_query` сейчас внутри себя зовёт `interrupt()`, поэтому LLM-планирование
нельзя выполнить вне работающего графа. Выделяем чистый хелпер:

```python
async def plan_narrowing(
    candidates: list[dict], query: IdentityQuery, locale: str
) -> dict:
    """Промпт → LLM → парс JSON → {attribute, question, options}. Без interrupt."""
```

`narrow_query` начинает звать `plan_narrowing(...)`, а затем `interrupt(...)`.
Внешнее поведение графа не меняется; появляется тестируемая точка для аспекта №3.
Остальные LLM-узлы уже вызываемы напрямую — их не трогаем.

## Запуск (opt-in маркеры)

- Регистрируем маркеры `eval` и `live` в `pyproject.toml`
  (`[tool.pytest.ini_options].markers`).
- В `addopts` добавляем `-m "not eval"` — обычный `pytest` остаётся быстрым,
  зелёным и не требует сетевых вызовов/ключей.
- Запуск оценок: `pytest -m eval`. По умолчанию (Ollama + `gpt-oss`) работает
  локально и бесплатно — ключи не нужны.
- Живой end-to-end: `pytest -m "eval and live"` (двойной opt-in).
- `tests/evals/conftest.py` делает автоскип, только если запуск объективно
  невозможен: выбран провайдер, требующий ключа (`anthropic`/`openai`), но ключ
  не задан, **либо** выбран `ollama`, но сервер по `OLLAMA_BASE_URL` недоступен.
  Цель — чтобы `pytest -m eval` в любой конфигурации либо отрабатывал, либо
  аккуратно скипался, но не падал по причине окружения.

## Поток данных (на примере faithfulness)

1. Тест читает `data/pages/jane_doe_github.md` и соответствующую запись из `goldens.json`.
2. Вызывает реальный `extract_profile_from_page(full_name=..., markdown=<страница>, ...)`.
3. Строит `LLMTestCase(input=<контекст-запрос>, actual_output=<профиль как текст>,
   retrieval_context=[<markdown страницы>])`.
4. Прогоняет `FaithfulnessMetric(model=LangChainJudge())` + `HallucinationMetric`
   + GEval; `assert_test(...)` падает, если порог не достигнут.

## Обработка ошибок и недетерминизм

- LLM недетерминирован: пороги метрик ставим консервативно (например 0.6–0.7),
  тесты помечены как медленные/опциональные. Локальный `gpt-oss` слабее облачных
  моделей — пороги учитывают это; при необходимости судью можно временно
  переключить на более сильную модель сменой `LLM_PROVIDER`/`LLM_MODEL`.
- Окружение не готово (нет ключа для cloud-провайдера или недоступна Ollama) →
  скип, не падение.
- Живой e2e зависит от сети и живых сайтов: один кейс, толерантный порог,
  явно задокументирован как best-effort.

## Зависимости

В dev-группу `[dependency-groups].dev` в `pyproject.toml` добавляем `deepeval`.
Тянет заметное число транзитивных пакетов, но это только dev — на прод-образ и
рантайм приложения не влияет.

## DEV.md — структура

**A. Оценка качества через DeepEval**
- что это и зачем opt-in (медленно, недетерминизм; по умолчанию бесплатно через
  локальную Ollama, но требует запущенного Ollama-сервера);
- как запускать: `pytest -m eval`, `pytest -m "eval and live"`; дефолтный
  локальный путь (Ollama + `gpt-oss`) и как переключить судью на cloud-провайдер;
- структура `tests/evals/`, роль судьи и голден-данных;
- как добавить новый голден (положить `.md` в `data/pages/` + запись в `goldens.json`);
- конфиг судьи (через `Settings`/`app/llm.py`), пороги, как читать падения метрик;
- заметка про стоимость и телеметрию (`DEEPEVAL_TELEMETRY_OPT_OUT=1`).

**B. Как менять граф**
- модель узел/ребро, что где лежит (`state.py` / `nodes.py` / `build.py` / `prompts.py`);
- пошаговые рецепты:
  - добавить узел (функция-нода → `add_node` → рёбра);
  - добавить условное ребро/роутер (`route_*` + `add_conditional_edges`);
  - изменить маршрутизацию;
  - добавить поле состояния (`PeopleSearchState`);
  - добавить `interrupt` (контракт payload ↔ resume);
- Mermaid-диаграмма текущего графа;
- чек-лист «после изменения графа обнови эти тесты»
  (`tests/test_graph_flow.py` и соответствующие `tests/evals/*`).

## Вне области (YAGNI)

- Не интегрируем Confident AI / облачные дашборды DeepEval.
- Не строим CI-pipeline для eval-тестов (только локальный opt-in запуск).
- Не добавляем синтетическую генерацию голден-датасетов.

## Критерии готовности

- `pytest` (без маркера) проходит как прежде, без сетевых вызовов.
- `pytest -m eval` на дефолтной конфигурации (Ollama + `gpt-oss`, без ключей)
  запускает 4 группы метрик и проходит на подготовленных голден-данных; если
  окружение не готово (cloud-провайдер без ключа или недоступная Ollama) — скип.
- `DEV.md` содержит оба раздела, команды запуска проверены, Mermaid-диаграмма
  отражает текущий граф.
