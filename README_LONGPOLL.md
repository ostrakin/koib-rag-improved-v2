# KOIB RAG — VK Long Poll бот (аддон)

Бот на **VK Bots Long Poll API** поверх готового индекса КОИБ. Переиспользует
ваш `RAGPipeline`, `VKBotService`, GigaChat-клиент и перенос индекса —
существующие файлы не меняются.

## Что внутри

| Файл                              | Назначение                                              |
|-----------------------------------|---------------------------------------------------------|
| `run_longpoll.py`                 | Точка входа (`python run_longpoll.py`)                   |
| `src/vk_longpoll.py`              | Движок Long Poll: опрос VK, дедуп, диспетчеризация       |
| `src/vk_menu_bot.py`             | Меню-навигация поверх `VKBotService`                    |
| `src/vk_keyboards.py`            | Клавиатуры (главное меню, модели, FAQ) и тексты         |
| `src/vk_faq.py`                  | Частые вопросы (кнопки → RAG)                            |
| `src/vk_prompt.py`               | Усиленный системный промпт GigaChat (опционально)       |
| `.env.longpoll.example`           | Пример конфигурации                                     |
| `deploy/koib-vk-longpoll.service` | systemd-юнит                                            |
| `deploy/run_longpoll.sh`          | Скрипт запуска с оптимальным окружением                 |
| `docs/VK_LONGPOLL_SETUP.md`       | Полная инструкция: структура, промпт, установка, перенос|

## Установка файлов

Скопируйте содержимое аддона в корень проекта, сохранив пути (`src/…`,
`deploy/…`, `docs/…`, корень). Существующие файлы не перезаписываются.

## Быстрый старт

```bash
# зависимости (если ещё не установлены)
pip install -r requirements-torch-cpu.txt
pip install -r requirements-server.txt

# конфигурация
cp .env.longpoll.example .env
nano .env           # GIGACHAT_CREDENTIALS, VK_ACCESS_TOKEN, VK_GROUP_ID

# индекс (перенос с индексатора)
python -m tools.index_transfer unpack --archive index_bundle.zip --output-dir ./output

# запуск
python run_longpoll.py
```

В VK: Сообщество → Управление → Работа с API → **Long Poll API** включить
(v5.131), тип события **`message_new`**; токен сообщества со scope
**messages + manage**.

Подробности, дерево меню, разбор промпта, перенос и systemd —
в `docs/VK_LONGPOLL_SETUP.md`.
