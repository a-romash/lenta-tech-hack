import hashlib
import os
from pathlib import Path

import jwt
import datetime
import io
import csv
import random

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
app = FastAPI()

env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)
# ---------- Конфигурация ----------
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY не задан в .env файле")

ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60

# Хеш пароля "admin" (SHA-256). Чтобы задать свой пароль, выполните:
# python -c "import hashlib; print(hashlib.sha256(b'ваш_пароль').hexdigest())"
PASSWORD_HASH = os.getenv("PASSWORD_HASH")
if not PASSWORD_HASH:
    raise ValueError("PASSWORD_HASH не задан в .env файле")

security = HTTPBearer()

# ---------- Вспомогательные функции ----------
def create_token() -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {"exp": expire, "sub": "user"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------- Ручки API ----------
@app.post("/api/login")
async def login(password: str = Form(...)):
    """Проверка пароля и выдача JWT-токена."""
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    if pwd_hash != PASSWORD_HASH:
        raise HTTPException(status_code=401, detail="Wrong password")
    token = create_token()
    return {"access_token": token, "token_type": "bearer"}

@app.post("/api/process")
async def process_video(
    video: UploadFile = File(...),
    token_payload: dict = Depends(verify_token)
):
    """
    Принимает видео, вызывает мок-нейросеть и отдаёт CSV.
    Токен проверяется автоматически через зависимость verify_token.
    """
    # Заглушка: генерация CSV с псевдо-результатами
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["frame", "timestamp", "detection"])
    for i in range(1, 11):
        writer.writerow([
            i,
            f"{i * 0.1:.1f}",
            random.choice(["cat", "dog", "car", "person"])
        ])
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=result.csv"}
    )

# ---------- Статика (должна быть после объявления API) ----------
app.mount("/", StaticFiles(directory="static", html=True), name="static")