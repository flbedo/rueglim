# Rueglim

Прототип объяснимого сравнения двух анкет для знакомств. Алгоритм находится
в `matching.py`, описание русскоязычной анкеты — в `questions.json`.

## Запуск

Нужен Python 3.10 или новее. Демонстрация и тесты используют только стандартную
библиотеку: ключи, внешние сервисы и скачивание моделей не нужны.

```bash
python3 matching.py
python3 -m unittest discover -s tests -v
```

## Использование

```python
from matching import HardPreferences, UserProfile, compare_users

a = UserProfile(
    user_id="a", age=28, gender="female", city="moscow",
    relationship_goal="serious_relationship",
    hard_preferences=HardPreferences(accepted_genders=frozenset({"male"})),
    structured={"home_outdoor": 40},
)
b = UserProfile(
    user_id="b", age=30, gender="male", city="moscow",
    relationship_goal="serious_relationship",
    hard_preferences=HardPreferences(accepted_genders=frozenset({"female"})),
    structured={"home_outdoor": 60},
)
result = compare_users(a, b, cross_fn=None)
print(result.final_score, result.coverage)
```

Сначала проверяются взаимные жёсткие ограничения: возраст, пол, город,
цель отношений и явно заданные ограничения образа жизни. Если они пройдены,
сравниваются структурированные ответы, теги, эмбеддинги и текстовые ответы.
Отсутствующие данные пропускаются; веса доступных групп пересчитываются.
`final_score` и `coverage` лежат в диапазоне 0–1. Низкое покрытие означает,
что итог опирается на небольшую часть данных.

## Необязательные модели

Для встроенных `embed()` и `cross_score()` установите
[sentence-transformers](https://pypi.org/project/sentence-transformers/)
в отдельное виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install sentence-transformers
```

При первом вызове скачиваются публичные модели Hugging Face:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` для эмбеддингов;
- `cross-encoder/ms-marco-MiniLM-L-6-v2` для текстового сравнения.

По умолчанию `compare_users()` использует `cross_score()`; `cross_fn=None`
отключает эту группу. Генерация эмбеддингов включается отдельно через
`auto_generate_embeddings=True`. Можно передать собственные `embed_fn`
и `cross_fn`, а готовые векторы хранить в `UserProfile.embeddings`.

## Состояние проекта

Это самостоятельное ядро алгоритма и схема анкеты. Интерфейс, сервер,
хранение анкет и преобразование ответов анкеты в `UserProfile` пока отсутствуют.
Например, `birth_date` нужно преобразовать в возраст, а `city_code` — в `city`.
Текст интересов также требует отдельного преобразования в взвешенные теги.

Веса и бонусы заданы эвристически и требуют проверки на реальных результатах.
Итоговый балл не является вероятностью успешных отношений. Встроенный
cross-encoder оценивает поисковую релевантность и не обучен предсказывать
совместимость людей; качество на русскоязычных анкетах требует отдельной оценки.

Проверка перед публикацией: тесты и демонстрация на Python 3.14.10.
Реальные модели и совместимость их зависимостей в этой проверке не запускались.

## Данные и секреты

`questions.json` содержит формулировки вопросов, а не заполненные анкеты.
Анкеты в демонстрации синтетические. Не добавляйте в репозиторий реальные
анкеты, медиа, токены, файлы `.env`, приватные ключи и кэши моделей.
Локальные конфигурации, секреты и данные исключены через `.gitignore`.
