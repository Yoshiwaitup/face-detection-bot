import os
import logging
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import cv2
import numpy as np
import io

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Конфигурация
BOT_TOKEN = "7419597367:AAF1wOt2sH57hBrToIhPbNYB51CxYk9vtW8"
API_URL = 'http://localhost:8000/detect_faces/'

# Инициализация бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    welcome_text = """
👋 Привет! Я бот для распознавания лиц на фотографиях.

🎓 Я являюсь частью университетского проекта по компьютерному зрению и машинному обучению.

🤖 Что я умею:
• Находить лица на фотографиях
• Определять их точное расположение
• Отмечать найденные лица рамками
• Показывать уровень уверенности в распознавании

📸 Просто отправь мне фотографию, и я:
1. Проанализирую её
2. Найду все лица
3. Отмечу их на изображении
4. Верну обработанное фото

ℹ️ Все данные сохраняются в базе данных для исследовательских целей.

🚀 Давай начнем! Отправь мне любую фотографию с людьми.
"""
    await message.reply(welcome_text)

@dp.message_handler(commands=['about'])
async def send_about(message: types.Message):
    about_text = """
🎯 О проекте:

Этот бот разработан в рамках университетского проекта по изучению технологий:
• FastAPI
• Computer Vision
• Machine Learning
• PostgreSQL
• Telegram Bot API

👨‍💻 Проект демонстрирует практическое применение:
• Распознавания образов
• Работы с базами данных
• Асинхронного программирования
• REST API

📚 Используемые технологии:
• Python
• OpenCV
• Roboflow
• PostgreSQL
• aiogram

🔒 Конфиденциальность:
Все данные используются только в исследовательских целях.
"""
    await message.reply(about_text)

@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    try:
        processing_msg = await message.reply("🔍 Обрабатываю изображение...\nЭто может занять несколько секунд.")
        photo = message.photo[-1]
        
        # Скачиваем фото
        file_info = await bot.get_file(photo.file_id)
        file_path = file_info.file_path
        
        # Создаем временную директорию
        os.makedirs('temp', exist_ok=True)
        temp_path = f'temp/{photo.file_id}.jpg'
        
        # Скачиваем файл
        await bot.download_file(file_path, temp_path)
        logger.info(f"Файл сохранен: {temp_path}")

        # Читаем изображение
        img = cv2.imread(temp_path)
        if img is None:
            raise ValueError("Не удалось прочитать изображение")

        # Отправляем на API
        async with aiohttp.ClientSession() as session:
            with open(temp_path, 'rb') as photo_file:
                form = aiohttp.FormData()
                form.add_field(
                    'file',
                    photo_file,
                    filename='photo.jpg',
                    content_type='image/jpeg'
                )
                
                logger.info("Отправляем запрос на API")
                async with session.post(API_URL, data=form) as response:
                    if response.status == 200:
                        faces_data = await response.json()
                        logger.info(f"Получены данные: {faces_data}")
                        
                        if faces_data['faces_found'] > 0:
                            # Рисуем рамки
                            for face in faces_data['faces']:
                                x = face['x']
                                y = face['y']
                                w = face['width']
                                h = face['height']
                                confidence = face['confidence']
                                
                                cv2.rectangle(
                                    img, 
                                    (x - w//2, y - h//2),
                                    (x + w//2, y + h//2),
                                    (0, 255, 0),
                                    2
                                )
                                
                                cv2.putText(
                                    img,
                                    f"{confidence:.2%}",
                                    (x - w//2, y - h//2 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2
                                )
                            
                            # Сохраняем и отправляем обработанное изображение
                            processed_path = f'temp/processed_{photo.file_id}.jpg'
                            cv2.imwrite(processed_path, img)
                            
                            with open(processed_path, 'rb') as processed_photo:
                                await message.reply_photo(
                                    processed_photo,
                                    caption=f"✅ Найдено лиц: {faces_data['faces_found']}\n"
                                            f"📊 Уровень уверенности: {min([face['confidence'] for face in faces_data['faces']]):.1%} - {max([face['confidence'] for face in faces_data['faces']]):.1%}"
                                )
                            
                            os.remove(processed_path)
                        else:
                            await message.reply("😕 На изображении не найдено лиц.\nПопробуйте другую фотографию!")
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка API: {error_text}")
                        await message.reply(f"Ошибка при обработке: {error_text}")

        # Удаляем временный файл
        os.remove(temp_path)
        await processing_msg.delete()
        
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}", exc_info=True)
        await message.reply(
            "❌ Произошла ошибка при обработке изображения.\n"
            "Пожалуйста, убедитесь, что:\n"
            "• Изображение содержит лица\n"
            "• Формат файла - JPEG или PNG\n"
            "• Размер файла не слишком большой\n\n"
            "Попробуйте другую фотографию!"
        )

@dp.message_handler(content_types=['text'])
async def handle_text(message: types.Message):
    if message.text.startswith('/'):
        await message.reply(
            "🤔 Извините, я не знаю такой команды.\n"
            "Доступные команды:\n"
            "/start - Начать работу\n"
            "/help - Получить помощь\n"
            "/about - О проекте\n\n"
            "Или просто отправьте мне фотографию! 📸"
        )

if __name__ == '__main__':
    logger.info("Бот запущен")
    executor.start_polling(dp, skip_updates=True) 