# mellat_premium_db.py
# سیستم کامل ذخیره‌سازی داده در PostgreSQL
# اجرا: python mellat_premium_db.py

import os
import logging
import json
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, JSON, Boolean,
    ForeignKey, Index, text, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# ----------------------------------------------------------------------
# تنظیمات لاگ
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mellat_db")

# ----------------------------------------------------------------------
# تنظیمات دیتابیس
# ----------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://mellat:securepassword123@localhost:5432/mellat_premium"
)

# ----------------------------------------------------------------------
# مدل‌های SQLAlchemy
# ----------------------------------------------------------------------
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False, index=True)
    device_id = Column(String(100), index=True)
    package_name = Column(String(100), default="com.yourapp.premium")
    is_premium = Column(Boolean, default=False)
    premium_expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    payments = relationship("Payment", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_users_user_device', 'user_id', 'device_id'),
    )

class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, unique=True, nullable=False, index=True)
    user_id = Column(String(50), ForeignKey('users.user_id', ondelete='SET NULL'), index=True)
    amount = Column(Integer, nullable=False)  # به ریال
    status = Column(String(20), nullable=False, default="pending")
    ref_id = Column(String(50), index=True)
    sale_order_id = Column(Integer)
    sale_reference_id = Column(String(50), index=True)
    gateway_response = Column(JSONB, nullable=True)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="payments")
    
    __table_args__ = (
        Index('ix_payments_status_created', 'status', 'created_at'),
        Index('ix_payments_user_status', 'user_id', 'status'),
    )

class AccessLog(Base):
    __tablename__ = "access_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), index=True)
    device_id = Column(String(100), index=True)
    access_token = Column(String(500), index=True)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    endpoint = Column(String(100))
    is_valid = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('ix_logs_user_token', 'user_id', 'access_token'),
        Index('ix_logs_created', 'created_at'),
    )

class AnalyticsSummary(Base):
    __tablename__ = "analytics_summary"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, unique=True, index=True)  # تاریخ بدون زمان
    total_revenue = Column(Integer, default=0)
    total_payments = Column(Integer, default=0)
    successful_payments = Column(Integer, default=0)
    failed_payments = Column(Integer, default=0)
    active_users = Column(Integer, default=0)
    new_users = Column(Integer, default=0)
    fraud_attempts = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ----------------------------------------------------------------------
# ایجاد دیتابیس
# ----------------------------------------------------------------------
def create_database():
    db_name = DATABASE_URL.split('/')[-1].split('?')[0]
    base_url = DATABASE_URL.rsplit('/', 1)[0] + '/postgres'
    
    try:
        conn = psycopg2.connect(base_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()
        if not exists:
            cur.execute(f"CREATE DATABASE {db_name} ENCODING 'UTF8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0")
            logger.info(f"دیتابیس `{db_name}` ایجاد شد.")
        else:
            logger.info(f"دیتابیس `{db_name}` قبلاً وجود دارد.")
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"خطا در ایجاد دیتابیس: {e}")

# ----------------------------------------------------------------------
# ایجاد جداول
# ----------------------------------------------------------------------
def create_tables(engine):
    Base.metadata.create_all(engine)
    logger.info("تمام جداول ایجاد شدند.")

# ----------------------------------------------------------------------
# مدیریت سشن
# ----------------------------------------------------------------------
engine = None
SessionLocal = None

def get_engine():
    global engine
    if engine is None:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
    return engine

def get_session():
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return SessionLocal()

# ----------------------------------------------------------------------
# توابع ذخیره‌سازی
# ----------------------------------------------------------------------
def save_user(user_id: str, device_id: str, package_name: str = "com.yourapp.premium"):
    session = get_session()
    try:
        user = session.query(User).filter_by(user_id=user_id).first()
        if not user:
            user = User(user_id=user_id, device_id=device_id, package_name=package_name)
            session.add(user)
            logger.info(f"کاربر جدید ذخیره شد: {user_id}")
        else:
            user.device_id = device_id
            user.package_name = package_name
            logger.info(f"کاربر به‌روزرسانی شد: {user_id}")
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"خطا در ذخیره کاربر: {e}")
    finally:
        session.close()

def save_payment(
    order_id: int,
    user_id: str,
    amount: int,
    status: str,
    ref_id: Optional[str] = None,
    sale_order_id: Optional[int] = None,
    sale_reference_id: Optional[str] = None,
    metadata: Optional[Dict] = None
):
    session = get_session()
    try:
        payment = Payment(
            order_id=order_id,
            user_id=user_id,
            amount=amount,
            status=status,
            ref_id=ref_id,
            sale_order_id=sale_order_id,
            sale_reference_id=sale_reference_id,
            metadata=metadata or {}
        )
        session.add(payment)
        session.commit()
        
        # به‌روزرسانی پریمیوم کاربر
        if status == "success":
            user = session.query(User).filter_by(user_id=user_id).first()
            if user:
                user.is_premium = True
                user.premium_expires_at = datetime.utcnow() + timedelta(days=365)
                session.commit()
        
        logger.info(f"پرداخت ذخیره شد: order_id={order_id}, status={status}")
    except Exception as e:
        session.rollback()
        logger.error(f"خطا در ذخیره پرداخت: {e}")
    finally:
        session.close()

def log_access(
    user_id: str,
    device_id: str,
    access_token: str,
    ip_address: str,
    user_agent: str,
    endpoint: str,
    is_valid: bool = True
):
    session = get_session()
    try:
        log = AccessLog(
            user_id=user_id,
            device_id=device_id,
            access_token=access_token,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            is_valid=is_valid
        )
        session.add(log)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"خطا در ثبت لاگ دسترسی: {e}")
    finally:
        session.close()

def update_daily_summary():
    session = get_session()
    try:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)  # ابتدای روز
        tomorrow = today + timedelta(days=1)

        summary = session.query(AnalyticsSummary).filter_by(date=today).first()
        if not summary:
            summary = AnalyticsSummary(date=today)
            session.add(summary)

        # محاسبه آمار
        payments = session.query(Payment).filter(
            Payment.created_at >= today,
            Payment.created_at < tomorrow
        ).all()

        successful = sum(1 for p in payments if p.status == "success")
        failed = sum(1 for p in payments if p.status == "failed")
        revenue = sum(p.amount for p in payments if p.status == "success")

        summary.total_revenue = revenue
        summary.total_payments = len(payments)
        summary.successful_payments = successful
        summary.failed_payments = failed
        summary.active_users = session.query(User).filter(User.is_premium == True).count()
        summary.new_users = session.query(func.count(User.id)).filter(
            func.date(User.created_at) == today.date()
        ).scalar() or 0

        session.commit()
        logger.info(f"خلاصه روزانه به‌روزرسانی شد: {today.date()}")
    except Exception as e:
        session.rollback()
        logger.error(f"خطا در به‌روزرسانی خلاصه: {e}")
    finally:
        session.close()

# ----------------------------------------------------------------------
# اجرای اولیه
# ----------------------------------------------------------------------
def init_db():
    create_database()
    engine = get_engine()
    create_tables(engine)
    logger.info("دیتابیس آماده استفاده است.")

# ----------------------------------------------------------------------
# اجرا
# ----------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    print("\nدیتابیس پریمیوم ملت با موفقیت راه‌اندازی شد.")
    print("جداول: users, payments, access_logs, analytics_summary")
    print("آماده استفاده در FastAPI، تحلیل داده و داشبورد.")