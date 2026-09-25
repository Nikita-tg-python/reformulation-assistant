# Reformulation Assistant

HTTP-сервіс для харчової R&D: технолог подає рецептуру і ціль (`remove_allergen`, `reduce_sugar`, `make_vegan`),
сервіс повертає заміни інгредієнтів з джерелами, нутрієнти до/після на 100 г і `trace` викликів інструментів.
Три частини: база знань (чанки з векторами в Postgres), RAG-пошук `/ask` з цитатами, агент `/reformulate`
з власним циклом tool calling (3 інструменти: `search_knowledge_base`, `lookup_product`, `calc_nutrition`).
Фронтенду немає, демо через Swagger UI на `/docs`.

Повна специфікація (схема БД, ендпоінти, ліміти агента, тести, критерії готовності, план на день): [docs/task.docx](docs/task.docx).

## Стек
Python 3.12, uv, FastAPI, Pydantic v2, pydantic-settings, uvicorn, asyncpg з ручним SQL, PostgreSQL 16 + pgvector
(`pgvector/pgvector:pg16`, hnsw + повнотекстовий GIN, гібридний пошук через RRF), ембединги `intfloat/multilingual-e5-small` (384) через ONNX Runtime без torch (не MiniLM зі спеки, бо корпус український),
LLM: Groq (за замовчуванням, `LLM_PROVIDER=groq`) або Gemini, Open Food Facts API, Docker Compose, pytest + pytest-asyncio + httpx, ruff.

## Структура
```
app/            main, config, db, schemas, embeddings, chunking, retrieval, ingest, errors, deps, logging_config
app/llm/        base.py (LLMClient, повтори), gemini.py, groq.py, fake.py (FakeLLM для тестів)
app/agent/      loop.py (цикл tool calling), pipeline.py (фіксований пайплайн, AGENT_MODE=pipeline),
                tools.py (3 інструменти + JSON-схеми), prompts.py
app/routers/    documents.py, ask.py, reformulate.py
data/corpus/    20 markdown-документів з frontmatter doc_id, title, doc_type
migrations/     001_init, 002_run_request_id, 003_fulltext (.sql); застосовуються на старті, без Alembic
tests/          test_chunking, test_calc_nutrition, test_agent_loop, test_pipeline, test_tools,
                test_llm_retries, test_retrieval, test_eval, test_api (+ conftest.py, fakes.py)
eval/           questions.jsonl (10 питань з doc_id) + run.py: recall@5 і MRR, vector vs hybrid
k8s/            бонус: kustomize для kind (Postgres, api, Job інжесту); k8s/secrets.env не комітити
.github/        workflows/ci.yml: ruff, pytest з pgvector, eval, збірка образу, kustomize
```

## Жорсткі правила
- Жодного LangChain, LlamaIndex, LangGraph чи іншого агентного/LLM-фреймворку. Цикл агента пишемо самі.
- LLM викликається тільки через інтерфейс `LLMClient` (`complete`, `complete_with_tools`) з `app/llm/base.py`.
  Провайдер обирається змінною `LLM_PROVIDER`. Жодних прямих викликів SDK провайдерів поза `app/llm/`.
- Тести тільки з `FakeLLM` і фейковим ембедером, без ключів і без інтернету. Живий LLM у тестах ніколи.
  Інтеграційні тести з БД пропускаються через `pytest.mark.skipif`, якщо немає `DATABASE_URL`.
- Нутрієнти рахує тільки `calc_nutrition` (чиста арифметика на Python), ніколи LLM і ніколи вручну в промпті.
- Єдиний формат помилок для всіх ендпоінтів: `{"error": {"code": "...", "message": "..."}}`.
- Секрети тільки зі змінних оточення через pydantic-settings. Файл `.env` ніколи не читати, не виводити
  і не комітити. Нові змінні додавати в `.env.example` з коментарем.
- Усі тіла запитів і відповідей описані Pydantic-моделями в `app/schemas.py`.

## Процес
- Працюємо блок за блоком за розділом «План на день» у специфікації. Один блок за раз.
- Після блоку зупинитися і дати контрольну точку для ручної перевірки. Наступний блок не починати без команди.
- Один коміт на блок, з читабельним повідомленням. Не комітити без підтвердження.
- Якщо рішення неочевидне (hnsw vs ivfflat, asyncpg vs ORM тощо), коротко пояснити чому.

## Команди
Хосту потрібні лише Docker і make (uv не встановлений; тести й ruff ідуть у контейнері).
```bash
cp .env.example .env        # вписати ключ LLM; .env не читати і не комітити
make up                     # збірка, api + db, чекає /health
make ingest                 # залити data/corpus/ (ідемпотентно)
make test / make eval        # pytest без ключів і інтернету / якість RAG (recall@5, MRR)
make lint                   # ruff check + format --check; make fmt виправляє
make logs / make down       # JSON-логи api / зупинка (дані лишаються)
make k8s-up / ingest-k8s    # kind: кластер, образ, apply -k (з інжестом) / повторний інжест
```
