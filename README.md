# Reformulation Assistant

[![CI](https://github.com/Nikita-tg-python/reformulation-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Nikita-tg-python/reformulation-assistant/actions/workflows/ci.yml)

HTTP-сервіс для харчової R&D. Технолог подає рецептуру й ціль: прибрати алерген, знизити цукор або зробити продукт веганським. Сервіс пропонує заміни інгредієнтів із посиланнями на внутрішні документи та Open Food Facts і показує нутрієнти на 100 г до та після.

Пет-проєкт на один робочий день під вакансію Backend & AI Engineer. Стек: Python 3.12, FastAPI, PostgreSQL + pgvector, RAG, агент з tool calling, Docker. Повна постановка задачі лежить у [docs/task.docx](docs/task.docx).

## Для кого і що вміє

- **Технолог R&D.** `POST /reformulate` повертає заміни, алергени до/після, нутрієнти до/після і `trace`: список усіх викликів інструментів агента.
- **Будь-хто в команді.** `POST /ask` відповідає на питання про інгредієнти лише з бази знань і наводить джерела. Якщо відповіді в документах немає, сервіс так і каже: «в базі знань немає даних».
- **Той, хто наповнює базу.** `POST /documents` або `make ingest` додають документи, повторний інгест того самого `doc_id` замінює чанки, а не дублює їх.

Фронтенду немає, для демо є Swagger UI на http://localhost:8000/docs.

## Архітектура

```mermaid
flowchart LR
    U[Клієнт / Swagger] --> API

    subgraph API[FastAPI]
        MW[middleware: request_id, JSON-логи]
        D[POST /documents]
        A[POST /ask]
        R[POST /reformulate]
        H[GET /health]
    end

    D --> CH[chunking: 400 токенів, перекриття 50]
    CH --> EMB[ембединги: multilingual-e5-small, ONNX Runtime]
    A --> RET[retrieval: hybrid = cosine hnsw + tsvector, RRF]
    RET --> EMB
    A --> LLM[LLMClient: Gemini або Groq]

    R --> AG[агент: loop ≤6 викликів LLM або pipeline 2–3, ≤60 с]
    AG --> LLM
    AG --> T1[search_knowledge_base]
    AG --> T2[lookup_product]
    AG --> T3[calc_nutrition: чиста арифметика]
    T1 --> RET
    T2 --> OFF[(Open Food Facts API)]

    subgraph PG[PostgreSQL 16 + pgvector]
        DOCS[(documents)]
        CHUNKS[(chunks: vector 384 hnsw, tsv GIN)]
        RUNS[(reformulation_runs: request, response, trace, status, duration_ms, request_id)]
    end

    EMB --> CHUNKS
    D --> DOCS
    RET --> CHUNKS
    R --> RUNS
```

- `app/llm/`: провайдер LLM сховано за інтерфейсом `LLMClient` з методами `complete` і `complete_with_tools`, обирається змінною `LLM_PROVIDER` (за замовчуванням `groq`: на free tier він стабільніший). `FakeLLM` використовується в тестах.
- `app/agent/`: цикл tool calling на ~75 рядків без фреймворків, три інструменти і промпт.
- `migrations/*.sql`: ідемпотентні міграції, застосовуються на старті під advisory lock.

## Запуск за 3 команди

Потрібні лише Docker і `make`. uv на машині не потрібен, усе працює в контейнерах.

```bash
cp .env.example .env   # вписати GROQ_API_KEY (безкоштовно, без картки: https://console.groq.com/keys)
make up                # збирає образ, піднімає api + db, чекає /health
make ingest            # заливає 20 документів із data/corpus/ у базу знань
```

Без `make`: `docker compose up -d --build`, потім `docker compose exec api python -m app.ingest data/corpus/`.

Решта команд: `make test` (pytest без ключів і без інтернету, разом з інтеграційними тестами на базі compose; у CI на GitHub Actions ті самі тести йдуть проти service-контейнера pgvector, без жодного пропущеного), `make lint`, `make logs`, `make down`, `make clean` (видаляє й дані). Повний список: `make help`.

`/health` і `/documents` працюють і без ключа LLM. Без ключа `/ask` і `/reformulate` повертають `503 llm_not_configured`.

## Ендпоінти

```bash
# стан сервісу, бази і провайдера LLM
curl -s localhost:8000/health
# {"status":"ok","db":"ok","llm_provider":"groq"}

# інгест одного документа (повтор із тим самим doc_id замінює чанки)
curl -s -X POST localhost:8000/documents -H 'Content-Type: application/json' -d '{
  "doc_id": "SPEC-099", "title": "Гороховий білок ізолят 80%", "doc_type": "ingredient_spec",
  "content": "## Функція в продукті\nПідвищує білок у рослинних йогуртах до 2-3 г на 100 г..."}'
# {"doc_id":"SPEC-099","chunks_created":1}

# RAG-відповідь із джерелами
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Чим замінити яйце в бісквіті?", "top_k": 5}'
# {"answer":"Для бісквіту рекомендовано замінити яйце аквафабою – 45 г на одне яйце; це забезпечує піну
#   і не вносить алергенів...","sources":[{"doc_id":"SPEC-010",...},{"doc_id":"TRIAL-005",...},{"doc_id":"SPEC-011",...}]}

# питання поза корпусом: відмова, а не вигадка
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Яка сьогодні погода в Києві?"}'
# {"answer":"В базі знань немає даних для відповіді на це питання.","sources":[]}

# агент переформулювання: три цілі на прикладі полуничного йогурту
BODY='"product_name":"Полуничний йогурт 2.5%","ingredients":[{"name":"молоко 2.5%","grams":800},{"name":"цукор","grams":90},{"name":"полуниця заморожена","grams":100},{"name":"закваска","grams":10}]'
curl -s -X POST localhost:8000/reformulate -H 'Content-Type: application/json' \
  -d "{$BODY,\"goal\":\"remove_allergen\",\"goal_params\":{\"allergen\":\"milk\"}}"
curl -s -X POST localhost:8000/reformulate -H 'Content-Type: application/json' \
  -d "{$BODY,\"goal\":\"reduce_sugar\",\"goal_params\":{\"percent\":30}}"
curl -s -X POST localhost:8000/reformulate -H 'Content-Type: application/json' \
  -d "{$BODY,\"goal\":\"make_vegan\",\"goal_params\":{}}"
```

Усі помилки мають один формат `{"error": {"code": "...", "message": "..."}}`. Помилки агента (`agent_timeout` 504, `agent_invalid_output` 502) додатково містять `trace`. Кожна відповідь має заголовок `X-Request-ID`, і той самий id є в JSON-логах і в `reformulation_runs`.

Перегляд запусків агента:

```bash
docker compose exec db psql -U postgres -d reformulation -c \
  "SELECT id, created_at, request->>'goal' AS goal, status, duration_ms, request_id,
          jsonb_array_length(trace) AS steps FROM reformulation_runs ORDER BY id DESC LIMIT 20"
```

## Як працює агент

Агент працює в одному з двох режимів. Режим задає змінна `AGENT_MODE` у `.env`.

| | `loop` (за замовчуванням) | `pipeline` |
|---|---|---|
| Хто викликає інструменти | модель сама (tool calling) | код, у фіксованому порядку |
| Викликів LLM на запит | до 6 (`AGENT_MAX_ITERATIONS`) | 2, або 3 з повтором |
| Типова тривалість на free tier | 15–60 с | 3–6 с |
| Що може зламатися | модель блукає й вичерпує ітерації, квоти free tier, 400 `output_parse_failed` на Groq | вибір моделі (заміна, для якої немає даних) |
| Коли обирати | сильна модель без жорстких квот | free tier (Groq 8000 токенів/хв, Gemini 5–20 запитів), демо |

Спільне для обох режимів:
- **Три інструменти.**

| Інструмент | Що робить |
|---|---|
| `search_knowledge_base(query, top_k)` | той самий пошук, що в `/ask` (hybrid за замовчуванням): один результат на документ (`top_k` ≤ 3), текст до 500 символів, для специфікацій також `nutrients_per_100g` |
| `lookup_product(name)` | Open Food Facts: нутрієнти, алергени (коди ЄС), інгредієнти; таймаут 5 с, кеш на один запуск, мережеві помилки повертаються як дані |
| `calc_nutrition(ingredients)` | зважене середнє kcal, білка, жиру, вуглеводів і цукру на 100 г, без LLM |

- **Ліміт часу 60 с** (`AGENT_TIMEOUT_SECONDS`). Перевищення дає 504 `agent_timeout` з уже зібраним trace.
- **Повтори запитів до провайдера** на 429 і 503 (з урахуванням `retry-after`), а на Groq ще й на 400 `output_parse_failed`: не більше 2 повторів.
- **Кожен запуск записується в `reformulation_runs`,** включно з невдалими.

### Режим `loop`: вільний цикл tool calling

Модель отримує рецептуру, ціль і опис інструментів, сама їх викликає, поки не готова відповісти, і повертає JSON. Щоб вкластися в квоти free tier:
- старі результати інструментів відправляються в історії стиснутими, повністю — лише в наступному виклику;
- формат відповіді описано в промпті компактно, а не згенерованою JSON-схемою;
- для Groq `gpt-oss` стоїть `reasoning_effort=low`.

Однаковий виклик двічі поспіль не виконується: агент примусово просить фінальну відповідь. Кожен запис trace містить `history_tokens`, щоб було видно, де впираємося в ліміт.

### Режим `pipeline`: фіксований порядок, 2 виклики LLM

Запасний варіант зі специфікації для випадку, коли вільний цикл нестабільний.
1. **Код** шукає в базі знань за ціллю, за кожним інгредієнтом (його власна специфікація) і «заміну для» кожного інгредієнта.
2. **LLM №1** обирає заміни, по одному інгредієнту на рядок; `original: null` означає добавку. Для кожного інгредієнта модель вказує, звідки взяти цифри.
3. **Код** перевіряє кожне число за джерелом: для специфікації — за її розділом «Нутрієнти на 100 г» (з урахуванням варіантів, як-от SPEC-009 А/Б), для решти шукає в Open Food Facts за англійською назвою. Число без джерела не приймається. Якщо даних бракує або вибір невалідний, LLM отримує другий шанс із переліком проблем.
4. **Код** двічі викликає `calc_nutrition`: для вихідного рецепта і для нового.
5. **LLM №2** пише фінальний JSON.

Модель у цьому режимі не викликає інструментів, лише повертає JSON. Тому збою розбору tool calls на Groq тут немає. Разом не більше 3 викликів LLM. Припущення коду (наприклад, «вихідна закваска — варіант А SPEC-009») записуються в trace.

### Перевірка відповіді (обидва режими)

Відповідь приймається, лише якщо:
- кожне джерело в `sources` справді повернули інструменти в цьому запуску, тому вигаданий `SPEC-014` не пройде;
- нутрієнти «до» пораховані `calc_nutrition` на вихідному рецепті (ті самі грами), а «після» — окремим викликом, без інгредієнтів із порожніми нутрієнтами;
- заміна без джерел має `confidence: "low"`, і для неї автоматично додається попередження в `warnings`;
- алергени процитованого продукту з OFF є в `allergens_after`, цільовий алерген прибрано, а у веганській версії немає тваринних алергенів.

Інакше модель отримує текст помилки й один повтор, а друга невдача дає 502 `agent_invalid_output`.

**Живий trace** (режим `pipeline`, Groq `openai/gpt-oss-120b`, ціль `remove_allergen`, 3,3 с, 2 виклики LLM; пошуки скорочено):

```json
{"iteration": 1, "type": "tool_call", "call": "search_knowledge_base(query=\"рослинна заміна інгредієнтів з алергеном milk, …\", top_k=3)", "result": ["TRIAL-004 (0.9057)", "SPEC-001 (0.9038)", "SPEC-006 (0.902)"]}
{"iteration": 1, "type": "tool_call", "call": "search_knowledge_base(query=\"закваска\", top_k=3)", "result": ["GUIDE-002 (0.8428)", "SPEC-009 (0.84)", "SPEC-002 (0.836)"]}
{"iteration": 1, "type": "llm_call", "purpose": "choose"}
{"iteration": 1, "type": "choice", "substitutions": ["молоко 2.5% -> кокосове молоко 2.5% жиру (800 g) ['SPEC-002']", "закваска -> закваска йогуртова, рослинна DVS (варіант Б) (10 g) ['SPEC-009']"]}
{"iteration": 1, "type": "assumption", "ingredient": "закваска", "note": "nutrients from SPEC-009, variant 1 of 2"}
{"iteration": 2, "type": "tool_call", "call": "lookup_product(name=\"frozen strawberries\")", "result": {"source_id": "OFF:9551014450028", "product_name": "frozen strawberry", "allergens": []}}
{"iteration": 2, "type": "tool_call", "call": "calc_nutrition(ingredients=[{\"name\": \"молоко 2.5%\", \"grams\": 800.0, …}])", "result": {"per_100g": {"kcal": 78.2, "protein_g": 2.4, "fat_g": 2.1, "carbs_g": 13.7, "sugar_g": 13.7}}}
{"iteration": 2, "type": "tool_call", "call": "calc_nutrition(ingredients=[{\"name\": \"кокосове молоко 2.5% жиру\", \"grams\": 800.0, …}])", "result": {"per_100g": {"kcal": 62.1, "protein_g": 0.3, "fat_g": 2.1, "carbs_g": 11.8, "sugar_g": 10.8}}}
{"iteration": 2, "type": "llm_call", "purpose": "final"}
{"iteration": 2, "type": "final_answer"}
```

Відповідь: молоко → кокосове молоко 2.5% жиру (`SPEC-002`), молочна закваска → рослинна DVS, варіант Б (`SPEC-009`); алергени `["milk"]` → `[]`; kcal 78.2 → 62.1, білок 2.4 → 0.3 г на 100 г. У warnings: втрата білка понад 30%, зміна текстури, потреба в стабілізаторах.

### Чесно про стан живого агента

- **`pipeline` на Groq:** кожна з трьох цілей пройшла наживо зі статусом 200 за 3–4 с і з 2 викликами LLM (remove_allergen, make_vegan, reduce_sugar), але в різних прогонах, не всі в одному. У `reduce_sugar` модель замінила весь цукор на ER-ST: цукор упав на 66% замість 30%, а 9% еритриту перевищує межу 8% зі SPEC-007. У `make_vegan` з вівсяним напоєм модель не вказала глютен. Алергени й доменні ліміти зі специфікацій код поки не перевіряє, бо вони не зберігаються окремими полями (див. «Що б зробив далі»).
- **`loop` на free tier в 6 ітерацій не вкладається.** На Groq заважають 8000 токенів/хв і 400 `output_parse_failed` (модель видає «роздуми» текстом замість виклику інструмента). На Gemini — квоти (5 запитів/хв для `gemini-3.5-flash`, 20 для `gemini-3.8-flash`) і 503 «high demand». Діагностичний прогін на `gemini-3.1-flash-lite` з лімітом 10 ітерацій дав 200 за 8 кроків, і модель замінила також молочну закваску.
- **Промпт агента дописано:** цілі в термінах R&D, доменні правила (приховані джерела алергену, новий алерген у warnings, втрата білка понад 30%) і приклад відповіді. У живих прогонах модель замінює і молочну закваску, а не лише молоко.
- **`/ask` на живій моделі працює:** Groq відповідає за 1–2 с, з джерелами, а на питання поза корпусом чесно відмовляє.

## Kubernetes (kind)

Маніфести лежать у [k8s/](k8s/) і збираються через kustomize, без Helm. Там Namespace, StatefulSet Postgres з PVC на 1 Гб і headless Service, Deployment api з однією реплікою і ClusterIP Service, а також Job для інжесту. ConfigMap і Secret генерує `kustomization.yaml`. Потрібні [kind](https://kind.sigs.k8s.io/) і `kubectl`.

```bash
make k8s-secrets                 # k8s/secrets.env: плейсхолдери + ключі LLM з локального .env
kind create cluster --name reformulation
docker build -t reformulation-assistant:local .
kind load docker-image reformulation-assistant:local --name reformulation
kubectl apply -k k8s/            # разом з Job інжесту: 20 документів
kubectl -n reformulation rollout status deployment/api --timeout=300s
kubectl -n reformulation port-forward svc/api 8000:8000   # в окремому терміналі
curl -s localhost:8000/health
# {"status":"ok","db":"ok","llm_provider":"groq"}
```

Те саме однією командою: `make k8s-up`. Повторний інжест: `make ingest-k8s`. Прибрати все: `make k8s-down`.

- **Секрети.** `secretGenerator` читає `k8s/secrets.env`, а цей файл у `.gitignore`. `make k8s-secrets` створює його з [k8s/secrets.env.example](k8s/secrets.env.example), де лише плейсхолдери, і дописує `GROQ_API_KEY` та `GEMINI_API_KEY` з `.env`, нічого не виводячи. Напряму з `../.env` kustomize читати не дає, бо файли поза `k8s/` він відхиляє. До того ж так у кластер потрапляють лише ключі, без решти `.env`. Несекретні налаштування лежать у ConfigMap, серед них `AGENT_MODE=pipeline`. Імена ConfigMap і Secret мають хеш вмісту, тож після зміни значення Deployment перекочується сам.
- **Інжест як Job, а не `kubectl exec`.** Спершу був простіший варіант з `exec` у под api, але інжест завантажує власну копію ONNX-моделі (пік близько 1.5 ГіБ). Разом з api це перевищило ліміт у 2 ГіБ, і OOM-killer вбив api. Job отримує свій под і свою пам'ять. Інжест ідемпотентний, тому Job запускається при першому `apply`. Після завершення Job видаляється через 10 хвилин, бо специфікація Job незмінна і стара Job заважала б наступному `apply`.
- **Ресурси.** Api в compose займає близько 1.1 ГіБ, на старті до 1.5 ГіБ, тому requests 1 ГіБ, limits 2 ГіБ. З лімітом 1 ГіБ под падав з OOMKilled ще на старті.
- **Проби на `/health`.** Коли база недоступна, `/health` повертає 503. Readiness прибирає под із Service, liveness перезапускає його приблизно через 30 с. Перевірено так: після `kubectl scale statefulset/postgres --replicas=0` api перезапустився через ~25 с, а після повернення бази піднявся сам, і дані в PVC збереглися. Компроміс: перезапуск api базу не лагодить, тож окремий `/livez`, який не залежить від бази, був би чистішим. startupProbe дає до 60 с на завантаження моделі й міграції, а initContainer чекає на Postgres.

## Рішення і відхилення від спеки

- **Модель ембедингів: `intfloat/multilingual-e5-small`, а не `all-MiniLM-L6-v2`.** Корпус український, а MiniLM англомовний: на 8 контрольних питаннях правильний документ у топ-3 він знаходив у 4 випадках, e5 — у 8. MiniLM до того ж обрізає вхід на 256 токенах, менше за чанк на 400. Розмірність та сама, 384, тому схема БД не змінилася.
- **ONNX Runtime замість sentence-transformers і torch.** Ті самі ваги й та сама нарізка, вектори збігаються з sentence-transformers (cos = 1.000000 на всьому корпусі). Образ зменшився з 2.8 ГБ до ~1.3 ГБ.
- **«Токени» в нарізці — це токени токенайзера моделі, а не слова.** Ліміт моделі теж рахується в токенах. Для української одне слово дає ~2 токени в e5 і ~5 у MiniLM.
- **asyncpg з ручним SQL.** Запитів мало, і вони специфічні для pgvector (`<=>`, `ON CONFLICT`). ORM тут був би зайвим шаром.
- **hnsw, а не ivfflat.** hnsw можна створити на порожній таблиці, і він коректно росте при вставках. ivfflat рахує кластери під час створення індексу, тож на старті був би непридатний.
- **Друга міграція** `002_run_request_id.sql` зв'язує запуски агента з request_id у логах.
- **Гібридний пошук (`SEARCH_MODE=hybrid`, за замовчуванням).** Векторний топ-20 і повнотекстовий топ-20 (`websearch_to_tsquery('simple', …)`, індекс GIN, міграція `003_fulltext.sql`) зливаються через Reciprocal Rank Fusion з k=60. Конфігурація `simple` без стемінгу, бо українського словника в Postgres немає, зате коди на кшталт SPEC-001 чи KS-2 збігаються дослівно. У `tsv` доданий `doc_id` з вагою A, бо чанк ніколи не містить коду власного документа: код трапляється лише в документах, що на нього посилаються. `score` у відповідях лишився косинусною подібністю, а порядок задає RRF. Порівняння hit@3 (top-3 чанки, e5-small, корпус із 20 документів):

  | Набір | vector | hybrid |
  |---|---|---|
  | 8 контрольних питань природною мовою | 8/8 | 8/8 |
  | 7 запитів-кодів (SPEC-001, SPEC-004, SPEC-009, TRIAL-003, KS-2, SG-3, ER-ST) | 3/7 | 4/7 |

  Обмеження RRF з рівними вагами: на запит «SPEC-004» повнотекстовий пошук ставить SPEC-004 першим, але документи, що його цитують, є в обох списках і після злиття обходять його.
- **Оцінка якості RAG: `make eval`.** [eval/run.py](eval/run.py) створює тимчасову базу, інжестить корпус і ставить 10 питань з [eval/questions.jsonl](eval/questions.jsonl): 8 сформульовані іншими словами, ніж у документах, 2 є кодами документів. Для кожного питання рахується, чи потрапив очікуваний документ у топ-5 документів (recall@5), і MRR@5. Обидва режими пошуку, поріг recall@5 ≥ 0.8. Той самий прогін іде в CI окремим job `eval` після тестів, з кешем моделі. Останній результат:

  | Режим | recall@5 | MRR@5 | коди: recall@5 |
  |---|---|---|---|
  | vector | 0.90 | 0.59 | 1/2 |
  | hybrid | 1.00 | 0.69 | 2/2 |

  На 8 смислових питаннях ранги в обох режимах однакові. Різницю дають коди: «SPEC-001» vector знаходить на 5-му місці, hybrid на 1-му. «TRIAL-003» vector не знаходить зовсім, hybrid знаходить на 5-му.
- **Структуровані дані специфікацій (міграція `004_structured_specs.sql`).** Під час інжесту [app/specs.py](app/specs.py) розбирає в кожній `ingredient_spec` секції «Нутрієнти на 100 г» і «Алергени». `documents.nutrients` (JSONB) зберігає варіанти продукту: «Аквафаба», «SG-3», «Варіант А». `documents.allergens` зберігає коди ЄС усіх варіантів разом. Алергени визначаються за номером пункту додатку II Регламенту 1169/2011, а не за словами, бо «Кокос не належить до горіхів» інакше дав би `nuts`. `search_knowledge_base` повертає `allergens` поруч із `nutrients_per_100g`. `calc_nutrition` звіряє вхідні нутрієнти інгредієнта, названого як специфікація в базі (усі слова назви є в заголовку або в назві варіанта), з `documents.nutrients` з допуском 0.05. Розбіжності йдуть у `data_mismatches` результату, а у фінальній відповіді стають попередженням у `warnings`, а не помилкою: зіставлення за назвою нечітке, а відхилена відповідь коштує повтору з ліміту в 6 викликів LLM. Перша ж жива перевірка знайшла справжню помилку. Після переходу на гібридний пошук (KAN-14) пайплайн і цикл брали специфікацію оригіналу як перший результат із нутрієнтами, і «закваска» отримала цифри кокосового молока, бо його текст згадує закваску. Тепер специфікація обирається за збігом назви із заголовком.
- **Postgres на хості слухає порт 5433,** бо 5432 часто зайнятий локальним Postgres.

## Що б зробив далі

1. **Доменні ліміти як дані.** Нутрієнти й алергени специфікацій уже структуровані й перевіряються кодом. Наступний крок: ліміти дозування (наприклад, еритрит ≤ 8% маси), бо в живому прогоні модель його перевищила. Крім того, алергени варто зберігати по варіантах: зараз SPEC-012 має `gluten`, хоча її варіант SG-3 безглютеновий.
2. **Точні коди в пошуку.** Гібридний пошук знаходить лише частину запитів-кодів (див. обмеження RRF вище). Варіанти: додавати код і назву документа в текст перед ембедингом (contextual chunk headers) або окремо шукати документи, чий `doc_id` точно збігся із запитом.
3. **Більший eval-набір.** 10 питань мало для висновків: одне питання змінює recall на 0.1. Варто 50+ питань від технологів і окремо оцінювати відповіді `/ask`: чи правильні цитати і чи модель відмовляє, коли відповіді в корпусі немає.
4. **Деплой у хмару.** Образ у registry, керований Postgres з pgvector, секрети в secret manager замість `k8s/secrets.env`, окремий `/livez` для liveness.
