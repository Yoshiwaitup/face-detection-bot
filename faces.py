from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
import os
from roboflow import Roboflow
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from datetime import datetime
import shutil
import io
import json
import base64
from fastapi.encoders import jsonable_encoder
import aiofiles
import logging
import getpass
from PIL import Image

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Подключение к базе данных
conn = psycopg2.connect(
    dbname="face_db",
    user="пользователь системы",  # Замените на имя вашего пользователя системы
    host="localhost",
    port="5432"
)
cursor = conn.cursor(cursor_factory=RealDictCursor)

# Создаем папку для хранения изображений если её нет
UPLOAD_DIR = "uploaded_images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Монтируем статические файлы
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

# Singleton для модели распознавания лиц
class FaceDetectionModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceDetectionModel, cls).__new__(cls)
            # Изменённая инициализация Roboflow
            rf = Roboflow(api_key="7ZG0FrfhVWiLbvPDBWTI")
            project = rf.workspace().project("face-detection-mik1i")
            cls._instance.model = project.version(21).model
        return cls._instance

    async def detect_faces(self, image_path: str):
        try:
            # Изменённый вызов предсказания
            return self.model.predict(image_path, confidence=40, overlap=30).json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка детекции лиц: {str(e)}")

face_detector = FaceDetectionModel()

@app.post("/detect_faces/")
async def detect_faces(file: UploadFile = File(...)) -> Dict:
    conn = None
    cur = None
    temp_image_path = None
    
    try:
        logger.info(f"""
        Получен файл:
        - Имя: {file.filename}
        - Тип контента: {file.content_type}
        """)

        # Читаем содержимое файла
        contents = await file.read()
        
        # Определяем формат файла
        file_ext = os.path.splitext(file.filename)[1].lower()
        content_type = file.content_type.lower()

        # Проверяем и конвертируем HEIC/HEIF
        if file_ext in ['.heic', '.heif'] or 'heic' in content_type or 'heif' in content_type:
            try:
                # Конвертируем HEIC в JPEG
                heic_image = Image.open(io.BytesIO(contents))
                jpeg_buffer = io.BytesIO()
                heic_image.save(jpeg_buffer, format='JPEG')
                contents = jpeg_buffer.getvalue()
                logger.info("HEIC/HEIF конвертирован в JPEG")
            except Exception as e:
                logger.error(f"Ошибка конвертации HEIC/HEIF: {str(e)}")
                raise HTTPException(400, detail="Ошибка при конвертации HEIC/HEIF формата")

        # Конвертируем изображение в массив numpy
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error(f"""
            Не удалось прочитать изображение:
            - Формат файла: {file_ext}
            - Тип контента: {content_type}
            - Размер данных: {len(contents)} байт
            """)
            raise HTTPException(400, detail="Не удалось прочитать изображение")

        # Сохраняем временный файл
        temp_image_path = f"temp_{file.filename.split('.')[0]}.jpg"
        cv2.imwrite(temp_image_path, img)

        # Детекция лиц
        detection_result = await face_detector.detect_faces(temp_image_path)
        
        # Подключаемся к базе данных
        conn = psycopg2.connect(
            dbname="face_db",
            user=getpass.getuser(),
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        try:
            # Сохраняем изображение в базу
            cur.execute(
                "INSERT INTO images (image_name, created_at) VALUES (%s, NOW()) RETURNING id",
                (file.filename,)
            )
            image_id = cur.fetchone()[0]

            # Сохраняем информацию о лицах
            faces = []
            for prediction in detection_result['predictions']:
                face = {
                    'x': int(prediction['x']),
                    'y': int(prediction['y']),
                    'width': int(prediction['width']),
                    'height': int(prediction['height']),
                    'confidence': float(prediction['confidence'])
                }
                faces.append(face)

                # Сохраняем каждое лицо в базу
                cur.execute("""
                    INSERT INTO faces (image_id, x, y, width, height, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (image_id, face['x'], face['y'], face['width'], face['height'], face['confidence']))

            # Подтверждаем транзакцию
            conn.commit()

            # Формируем ответ
            response_data = {
                'status': 'success',
                'image_id': image_id,
                'faces_found': len(faces),
                'image_size': {
                    'width': img.shape[1],
                    'height': img.shape[0]
                },
                'faces': faces
            }

            logger.info(f"Найдено лиц: {len(faces)}")
            
            return JSONResponse(content=response_data)

        except Exception as db_error:
            if conn:
                conn.rollback()
            logger.error(f"Ошибка базы данных: {str(db_error)}")
            raise HTTPException(500, detail=f"Ошибка базы данных: {str(db_error)}")

    except Exception as e:
        logger.error(f"""
        Ошибка обработки файла:
        - Тип ошибки: {type(e).__name__}
        - Сообщение: {str(e)}
        """)
        if conn:
            conn.rollback()
        raise HTTPException(500, detail=str(e))

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

@app.get("/images/{image_id}")
async def get_image(image_id: int):
    cursor.execute("SELECT image_name FROM images WHERE id = %s", (image_id,))
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    
    image_path = os.path.join(UPLOAD_DIR, result["image_name"])
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Файл изображения не найден")
    
    return FileResponse(image_path)

# Добавляем новый эндпоинт для получения статистики пользователя
@app.get("/user_statistics/{telegram_user_id}")
async def get_user_statistics(telegram_user_id: int):
    try:
        conn = psycopg2.connect(
            dbname="face_db",
            user="postgres",
            password="your_password",
            host="localhost"
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Получаем общую статистику пользователя
        cur.execute("""
            SELECT 
                COUNT(DISTINCT i.id) as total_images,
                COUNT(f.id) as total_faces,
                ROUND(AVG(faces_per_image), 2) as avg_faces_per_image,
                MAX(i.created_at) as last_upload
            FROM images i
            LEFT JOIN (
                SELECT image_id, COUNT(*) as faces_per_image 
                FROM faces 
                GROUP BY image_id
            ) f_count ON i.id = f_count.image_id
            LEFT JOIN faces f ON i.id = f.image_id
            WHERE i.telegram_user_id = %s
            GROUP BY i.telegram_user_id
        """, (telegram_user_id,))
        
        stats = cur.fetchone()

        # Получаем последние 5 загрузок
        cur.execute("""
            SELECT 
                i.image_name,
                COUNT(f.id) as faces_found,
                i.created_at
            FROM images i
            LEFT JOIN faces f ON i.id = f.image_id
            WHERE i.telegram_user_id = %s
            GROUP BY i.id
            ORDER BY i.created_at DESC
            LIMIT 5
        """, (telegram_user_id,))
        
        recent_uploads = cur.fetchall()

        cur.close()
        conn.close()

        return {
            "user_id": telegram_user_id,
            "statistics": stats,
            "recent_uploads": recent_uploads
        }

    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {str(e)}")
        raise HTTPException(500, detail=str(e))

# Не забудь создать таблицы в базе данных:
# CREATE TABLE images (id SERIAL PRIMARY KEY, image_name VARCHAR(255));
# CREATE TABLE faces (id SERIAL PRIMARY KEY, image_id INTEGER REFERENCES images(id), x INTEGER, y INTEGER, w INTEGER, h INTEGER);
