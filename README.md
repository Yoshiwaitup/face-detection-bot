## 📋 Требования

- Python 3.8+
- PostgreSQL
- Telegram Bot Token (получить у [@BotFather](https://t.me/BotFather))

## 🛠 Установка

1. Клонируйте репозиторий:
git clone https://github.com/your-username/face-detection-bot.git
cd face-detection-bot

2. Установите зависимости:
pip install -r requirements.txt

3. Создайте базу данных PostgreSQL:
createdb face_db
psql -d face_db -f init.sql

4. Настройте бота:
- Получите токен у [@BotFather](https://t.me/BotFather)
- Вставьте токен в `telegram_bot.py`

## 🚀 Запуск

1. Запустите API:
uvicorn faces:app --reload

2. В другом терминале запустите бота:
python telegram_bot.py

## 💡 Использование

1. Найдите бота в Telegram
2. Отправьте команду `/start` для начала работы
3. Отправьте фотографию
4. Получите результат с отмеченными лицами

## 📝 Команды бота

- `/start` - Начать работу с ботом
- `/help` - Получить справку
- `/about` - Информация о проекте

## 🗄 Структура проекта

face-detection-bot/
├── faces.py           # FastAPI сервер
├── telegram_bot.py    # Telegram бот
├── init.sql          # SQL схема базы данных
└── requirements.txt   # Зависимости проекта

## 🔧 Технологии

- FastAPI - веб-фреймворк
- OpenCV - компьютерное зрение
- PostgreSQL - база данных
- aiogram - Telegram Bot API
- Pillow - обработка изображений
- YOLOv5 - модель детекции лиц

## 📊 База данных

Проект использует две таблицы:
- `images` - информация о загруженных изображениях
- `faces` - данные о найденных лицах

## 🌐 Развертывание

версия с GitHub

## 🔄 Текущий статус

Проект находится на стадии прототипа и активно развивается. Планируется:
- Улучшение точности распознавания
- Оптимизация производительности
- Добавление новых функций
- Развертывание на production сервере
