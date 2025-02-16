-- Создаем таблицу для хранения информации о загруженных изображениях
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    image_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Создаем таблицу для хранения информации о найденных лицах
CREATE TABLE IF NOT EXISTS faces (
    id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES images(id),
    x INTEGER,
    y INTEGER,
    width INTEGER,
    height INTEGER,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);