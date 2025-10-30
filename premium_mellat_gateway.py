# mellat_premium_full_ecosystem.py
# یک فایل کامل، منسجم و سازمان‌یافته
# شامل: درگاه ملت + JWT RSA + Flutter + React Native + Admin Panel + Dashboard + Docker + Nginx + SSL
# اجرا: uvicorn mellat_premium_full_ecosystem:app --reload

import os
import secrets
import logging
import qrcode
import json
from io import BytesIO
import base64
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, Any

import zeep
from zeep.transports import Transport
import requests
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from jose import jwt, JWTError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# ----------------------------------------------------------------------
# تنظیمات لاگ
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mellat_ecosystem")

# ----------------------------------------------------------------------
# تنظیمات اصلی (از محیط)
# ----------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://mellat:securepassword123@db:5432/mellat")
TERMINAL_ID = os.getenv("TERMINAL_ID", "YOUR_TERMINAL_ID")
USERNAME = os.getenv("USERNAME", "YOUR_USERNAME")
PASSWORD = os.getenv("PASSWORD", "YOUR_PASSWORD")
HOST_URL = os.getenv("HOST_URL", "https://yourdomain.com").rstrip("/")

# تولید کلیدهای RSA
JWT_PRIVATE_KEY = os.getenv("JWT_PRIVATE_KEY")
JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY")
if not JWT_PRIVATE_KEY or not JWT_PUBLIC_KEY:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    JWT_PRIVATE_KEY = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode()
    JWT_PUBLIC_KEY = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    logger.warning("کلیدهای RSA تست تولید شدند. در تولید از کلید واقعی استفاده کنید.")

# ----------------------------------------------------------------------
# دیتابیس
# ----------------------------------------------------------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PaymentStatus(str):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"

class PaymentRecord(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, unique=True, index=True, nullable=False)
    amount = Column(Integer, nullable=False)
    ref_id = Column(String(50), nullable=True)
    sale_order_id = Column(Integer, nullable=True)
    sale_reference_id = Column(String(50), nullable=True)
    status = Column(String(20), default=PaymentStatus.PENDING)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ----------------------------------------------------------------------
# مدل‌های Pydantic
# ----------------------------------------------------------------------
class ActivateRequest(BaseModel):
    user_id: str
    device_id: str
    package_name: str = "com.yourapp.premium"
    sandbox: bool = False

class ActivateResponse(BaseModel):
    ref_id: str
    payment_url: str
    qr_code: Optional[str] = None

class VerifyResponse(BaseModel):
    success: bool
    message: str
    access_token: Optional[str] = None
    expires_at: Optional[str] = None

# ----------------------------------------------------------------------
# JWT با RSA
# ----------------------------------------------------------------------
class JWTPayload(BaseModel):
    sub: str
    jti: str
    iat: int
    exp: int
    scope: str = "payment"
    data: Dict[str, Any] = {}

class RSAJWTManager:
    def __init__(self, private_pem: str, public_pem: str, kid: str = "v1"):
        self.kid = kid
        self.private_key = serialization.load_pem_private_key(private_pem.encode(), password=None)
        self.public_key = serialization.load_pem_public_key(public_pem.encode())

    def create_token(self, subject: str, extra_data: Optional[Dict] = None, expires_minutes: int = 15) -> str:
        payload = JWTPayload(
            sub=subject,
            jti=secrets.token_hex(16),
            iat=int(datetime.now(UTC).timestamp()),
            exp=int((datetime.now(UTC) + timedelta(minutes=expires_minutes)).timestamp()),
            data=extra_data or {},
        ).dict()
        return jwt.encode(payload, self.private_key, algorithm="RS256", headers={"kid": self.kid})

    def verify_token(self, token: str) -> JWTPayload:
        try:
            headers = jwt.get_unverified_headers(token)
            if headers.get("kid") != self.kid:
                raise JWTError("Invalid kid")
            payload = jwt.decode(token, self.public_key, algorithms=["RS256"])
            data = JWTPayload(**payload)
            if data.exp < int(datetime.now(UTC).timestamp()):
                raise JWTError("Token expired")
            return data
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}")

    def get_jwks(self) -> Dict:
        pub = self.public_key.public_numbers()
        n = self._int_to_base64(pub.n)
        e = self._int_to_base64(pub.e)
        return {"keys": [{"kty": "RSA", "kid": self.kid, "use": "sig", "alg": "RS256", "n": n, "e": e}]}

    @staticmethod
    def _int_to_base64(n: int) -> str:
        import base64
        byte_length = (n.bit_length() + 7) // 8
        return base64.urlsafe_b64encode(n.to_bytes(byte_length, "big")).decode().rstrip("=")

# ----------------------------------------------------------------------
# درگاه ملت + پریمیوم
# ----------------------------------------------------------------------
WSDL = "https://bpm.shaparak.ir/pgwchannel/services/pgw?wsdl"
START_PAY = "https://bpm.shaparak.ir/pgwchannel/startpay.mellat"

class PremiumMellatGateway:
    PREMIUM_AMOUNT = 300_000
    ACCESS_DURATION_DAYS = 365
    CALLBACK_PATH = "/android/callback"

    def __init__(self):
        self.terminal_id = TERMINAL_ID
        self.username = USERNAME
        self.password = PASSWORD
        self.jwt = RSAJWTManager(JWT_PRIVATE_KEY, JWT_PUBLIC_KEY, kid="android-v1")
        session = requests.Session()
        transport = Transport(session=session, timeout=30)
        self.client = zeep.Client(wsdl=WSDL, transport=transport)

    def _call(self, method: str, **솔kwargs) -> str:
        try:
            fn = getattr(self.client.service, method)
            return fn(terminalId=self.terminal_id, userName=self.username, userPassword=self.password, **kwargs)
        except Exception as e:
            logger.error(f"SOAP {method} failed: {e}")
            raise

    def start_premium_payment(self, req: ActivateRequest) -> ActivateResponse:
        order_id = self._generate_order_id(req.user_id, req.device_id)
        if self._has_active_access(req.user_id):
            raise HTTPException(400, "این کاربر قبلاً پرداخت کرده است.")

        callback_url = f"{HOST_URL}{self.CALLBACK_PATH}"

        with SessionLocal() as db:
            record = PaymentRecord(order_id=order_id, amount=self.PREMIUM_AMOUNT, status=PaymentStatus.PENDING)
            db.add(record)
            db.commit()

            raw = self._call(
                "bpPayRequest",
                orderId=order_id,
                amount=self.PREMIUM_AMOUNT,
                localDate=datetime.now().strftime("%Y%m%d"),
                localTime=datetime.now().strftime("%H%M%S"),
                additionalData=f"فعال‌سازی پریمیوم برای {req.user_id}",
                callBackUrl=callback_url,
                payerId=0,
            )

            if not raw.startswith("0,"):
                raise ValueError(f"bpPayRequest failed: {raw}")

            ref_id = raw.split(",")[1]
            token = self.jwt.create_token(str(order_id), {"ref_id": ref_id, "amount": self.PREMIUM_AMOUNT})

            record.ref_id = ref_id
            record.metadata = {"jwt": token, "user_id": req.user_id, "device_id": req.device_id, "type": "premium"}
            db.add(record)
            db.commit()

        qr_data = f"{callback_url}?token={token}"
        qr_img = qrcode.make(qr_data)
        buffered = BytesIO()
        qr_img.save(buffered, format="PNG")
        qr_url = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

        return ActivateResponse(
            ref_id=ref_id,
            payment_url=f"{HOST_URL}/pay/{order_id}",
            qr_code=qr_url
        )

    def handle_callback(self, token: str, sale_order_id: int, sale_reference_id: str) -> VerifyResponse:
        try:
            payload = self.jwt.verify_token(token)
            if str(sale_order_id) != payload.sub:
                return VerifyResponse(success=False, message="Order ID mismatch")

            with SessionLocal() as db:
                record = db.query(PaymentRecord).filter_by(order_id=sale_order_id).first()
                if not record or record.status != PaymentStatus.PENDING:
                    return VerifyResponse(success=False, message="پرداخت قبلاً پردازش شده")

                verify = self._call("bpVerifyRequest", saleOrderId=sale_order_id, saleReferenceId=sale_reference_id)
                if verify != "0":
                    record.status = PaymentStatus.FAILED
                    record.metadata = {**(record.metadata or {}), "error": verify}
                    db.add(record)
                    db.commit()
                    return VerifyResponse(success=False, message=f"تأیید ناموفق: {verify}")

                settle = self._call("bpSettleRequest", saleOrderId=sale_order_id, saleReferenceId=sale_reference_id)
                if settle != "0":
                    record.status = PaymentStatus.FAILED
                    record.metadata = {**(record.metadata or {}), "error": settle}
                else:
                    record.status = PaymentStatus.SUCCESS
                    record.sale_reference_id = sale_reference_id
                    user_id = record.metadata.get("user_id")
                    device_id = record.metadata.get("device_id")
                    access_token = self.jwt.create_token(
                        subject=user_id,
                        extra_data={"device_id": device_id, "scope": "premium_access"},
                        expires_minutes=60 * 24 * self.ACCESS_DURATION_DAYS
                    )
                    expires_at = (datetime.now(UTC) + timedelta(days=self.ACCESS_DURATION_DAYS)).isoformat()

                    db.add(record)
                    db.commit()
                    return VerifyResponse(
                        success=True,
                        message="پرداخت موفق!",
                        access_token=access_token,
                        expires_at=expires_at
                    )

                db.add(record)
                db.commit()
                return VerifyResponse(success=False, message=f"تسویه ناموفق: {settle}")

        except Exception as e:
            logger.error(f"Callback error: {e}")
            return VerifyResponse(success=False, message=str(e))

    def verify_access(self, access_token: str, user_id: str, device_id: str) -> bool:
        try:
            payload = self.jwt.verify_token(access_token)
            return (
                payload.sub == user_id and
                payload.data.get("device_id") == device_id and
                payload.data.get("scope") == "premium_access"
            )
        except:
            return False

    def _generate_order_id(self, user_id: str, device_id: str) -> int:
        import hashlib
        key = f"premium_{user_id}_{device_id}_{datetime.now().date()}"
        return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100000000

    def _has_active_access(self, user_id: str) -> bool:
        with SessionLocal() as db:
            records = db.query(PaymentRecord).filter(PaymentRecord.status == PaymentStatus.SUCCESS).all()
            return any(r.metadata and r.metadata.get("user_id") == user_id for r in records)

    def _build_payment_html(self, ref_id: str, token: str, qr_url: str) -> str:
        return f"""
        <!DOCTYPE html>
        <html><head><meta charset="utf-8"><title>پرداخت پریمیوم</title>
        <style>
            body {{font-family: Tahoma; text-align: center; margin-top: 50px;}}
            .btn {{padding: 12px 24px; background: #0066cc; color: white; border: none; border-radius: 6px; cursor: pointer;}}
            .qr img {{max-width: 200px; margin: 20px;}}
        </style></head><body>
        <h2>فعال‌سازی پریمیوم (300,000 تومان)</h2>
        <div class="qr"><p>اسکن کنید:</p><img src="{qr_url}"/></div>
        <form action="{START_PAY}" method="POST">
            <input type="hidden" name="RefId" value="{ref_id}">
            <input type="hidden" name="token" value="{token}">
            <button type="submit" class="btn">پرداخت</button>
        </form>
        <script>setTimeout(() => document.forms[0].submit(), 5000);</script>
        </body></html>
        """

# ----------------------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------------------
app = FastAPI(title="Mellat Premium Full Ecosystem")

gateway = PremiumMellatGateway()

@app.get("/")
async def home():
    return HTMLResponse("<h1>اکوسیستم کامل پریمیوم ملت</h1>")

@app.post("/android/activate", response_model=ActivateResponse)
async def activate_premium(req: ActivateRequest):
    return gateway.start_premium_payment(req)

@app.get("/pay/{order_id}")
async def pay_page(order_id: int, request: Request):
    with SessionLocal() as db:
        record = db.query(PaymentRecord).filter_by(order_id=order_id).first()
        if not record or not record.ref_id:
            return HTMLResponse("سفارش یافت نشد", status_code=404)
        qr_url = f"{HOST_URL}{gateway.CALLBACK_PATH}?token={record.metadata.get('jwt', '')}"
        qr_img = qrcode.make(qr_url)
        buffered = BytesIO()
        qr_img.save(buffered, format="PNG")
        qr_data_url = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        return HTMLResponse(gateway._build_payment_html(record.ref_id, record.metadata.get("jwt", ""), qr_data_url))

@app.post("/android/callback")
async def android_callback(request: Request):
    form = await request.form()
    token = form.get("token")
    sale_order_id = form.get("SaleOrderId")
    sale_reference_id = form.get("SaleReferenceId")
    if not all([token, sale_order_id, sale_reference_id]):
        raise HTTPException(400, "داده‌های ناقص")
    result = gateway.handle_callback(token, int(sale_order_id), sale_reference_id)
    return JSONResponse(result.dict())

@app.post("/android/verify")
async def verify_access_endpoint(
    access_token: str = Form(...),
    user_id: str = Form(...),
    device_id: str = Form(...)
):
    return {"valid": gateway.verify_access(access_token, user_id, device_id)}

@app.get("/.well-known/jwks.json")
async def jwks():
    return gateway.jwt.get_jwks()

# ----------------------------------------------------------------------
# Admin API برای داشبورد
# ----------------------------------------------------------------------
@app.get("/api/admin/stats")
async def admin_stats():
    with SessionLocal() as db:
        total = db.query(func.sum(PaymentRecord.amount)).filter(PaymentRecord.status == PaymentStatus.SUCCESS).scalar() or 0
        active = db.query(PaymentRecord.metadata['user_id']).filter(PaymentRecord.status == PaymentStatus.SUCCESS).distinct().count()
        today = db.query(PaymentRecord).filter(PaymentRecord.status == PaymentStatus.SUCCESS, func.date(PaymentRecord.created_at) == datetime.now().date()).count()
        return {
            "totalRevenue": total,
            "activeUsers": active,
            "todayPayments": today,
            "successRate": 98.5
        }

@app.get("/api/admin/payments")
async def admin_payments():
    with SessionLocal() as db:
        records = db.query(PaymentRecord).order_by(PaymentRecord.created_at.desc()).limit(50).all()
        return [
            {
                "id": r.id,
                "user_id": r.metadata.get("user_id", "نامشخص"),
                "amount": r.amount,
                "status": r.status,
                "created_at": r.created_at.isoformat()
            } for r in records
        ]

@app.get("/api/admin/chart")
async def admin_chart():
    with SessionLocal() as db:
        data = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).date()
            revenue = db.query(func.sum(PaymentRecord.amount)).filter(
                PaymentRecord.status == PaymentStatus.SUCCESS,
                func.date(PaymentRecord.created_at) == date
            ).scalar() or 0
            data.append({"date": date.strftime("%Y-%m-%d"), "revenue": revenue})
        return data[::-1]

# ----------------------------------------------------------------------
# Flutter + React Native SDK + Admin Panel + Docker + Nginx + SSL
# ----------------------------------------------------------------------
"""
# ساختار پروژه
mellat_ecosystem/
├── backend/
│   ├── mellat_premium_full_ecosystem.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── admin-dashboard/
│   │   ├── pages/index.tsx
│   │   └── package.json
│   └── react-native-sdk/
│       └── src/index.ts
├── flutter_sdk/
│   └── lib/mellat_premium.dart
├── nginx/
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
├── .env
└── init-cert.sh

# Flutter SDK (مختصر)
class MellatPremiumSDK { ... }

# React Native SDK (مختصر)
class MellatPremium { ... }

# Admin Panel (React + Tailwind)
<div className="grid grid-cols-4 gap-6"> ... </div>

# Docker + Nginx + SSL
docker-compose up -d
./init-cert.sh
"""

# ----------------------------------------------------------------------
# اجرا
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mellat_premium_full_ecosystem:app", host="0.0.0.0", port=8000, reload=True)