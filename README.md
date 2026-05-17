# Инструкция по развертыванию на VPS

## Предварительные требования

- VPS с linux
- Установленный Docker и Docker Compose
- Доступ по SSH

## Быстрый старт

Шаг 1: Настройка переменных окружения
Создайте файл .env на основе примера:


```bash
cp .env.example .env
```
Сгенерируйте значения для .env:

Генерация секретного ключа:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```
Генерация хеша пароля:

```bash
python3 -c "import hashlib; print(hashlib.sha256(b'ваш_пароль'.hexdigest())"
```

Отредактируйте .env файл:

```bash
nano .env
```


Пример заполненного .env:

```env
SECRET_KEY=a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2
PASSWORD_HASH=8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918
```

Шаг 5: Запуск приложения
Вариант А: Через Docker Compose (рекомендуется)

```bash
docker-compose up -d --build
```
