#!/usr/bin/env python3
# =============================================================
# mellat_db_secure_init.py
# راه‌اندازی کامل، امن و بدون SQL Injection دیتابیس پریمیوم ملت
# نسخه: 1.1.0 | تاریخ: 2025-04-05
# اجرا: python mellat_db_secure_init.py
# =============================================================

import os
import sys
import re
import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path

# ----------------------------------------------------------------------
# تنظیمات لاگ
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("mellat_db_init")

# ----------------------------------------------------------------------
# تنظیمات دیتابیس
# ----------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/postgres"
)
DB_NAME_RAW = os.getenv("DB_NAME", "mellat_premium").strip()

# ----------------------------------------------------------------------
# اعتبارسنجی سخت‌گیرانه نام دیتابیس
# ----------------------------------------------------------------------
def validate_db_name(name: str) -> str:
    """
    اعتبارسنجی نام دیتابیس:
    - فقط حروف کوچک، اعداد و آندراسکور
    - حداکثر 63 کاراکتر
    - نباید با عدد شروع شود
    """
    pattern = r'^[a-z_][a-z0-9_]{0,62}$'
    if not re.match(pattern, name):
        logger.error(f"نام دیتابیس نامعتبر: {name}")
        logger.error("نام باید فقط شامل حروف کوچک، عدد و _ باشد و با حرف شروع شود.")
        sys.exit(1)
    return name

DB_NAME = validate_db_name(DB_NAME_RAW)

# ----------------------------------------------------------------------
# مسیر فایل SQL
# ----------------------------------------------------------------------
SQL_FILE = Path(__file__).parent / "mellat_db_secure.sql"

if not SQL_FILE.exists():
    logger.error(f"فایل SQL پیدا نشد: {SQL_FILE}")
    sys.exit(1)

# ----------------------------------------------------------------------
# تابع ایجاد دیتابیس (کاملاً امن با Identifier)
# ----------------------------------------------------------------------
def create_database_safely() -> bool:
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # استفاده از Identifier (کاملاً امن)
        cur.execute(
            sql.SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = {}").format(
                sql.Identifier(DB_NAME)
            )
        )
        exists = cur.fetchone()

        if not exists:
            # استفاده از Identifier برای CREATE DATABASE
            create_sql = sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(
                sql.Identifier(DB_NAME)
            )
            cur.execute(create_sql)
            logger.info(f"دیتابیس `{DB_NAME}` با موفقیت ایجاد شد.")
        else:
            logger.info(f"دیتابیس `{DB_NAME}` قبلاً وجود دارد.")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"خطا در ایجاد دیتابیس: {e}")
        return False

# ----------------------------------------------------------------------
# تابع اجرای اسکریپت SQL (امن)
# ----------------------------------------------------------------------
def run_sql_script() -> bool:
    # ساخت URL امن برای دیتابیس هدف
    base_url = DATABASE_URL.rsplit('/', 1)[0]
    target_url = f"{base_url}/{DB_NAME}"

    try:
        conn = psycopg2.connect(target_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        with open(SQL_FILE, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        cur.execute(sql_content)
        logger.info(f"اسکریپت SQL با موفقیت در `{DB_NAME}` اجرا شد.")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"خطا در اجرای اسکریپت SQL: {e}")
        return False

# ----------------------------------------------------------------------
# تابع تست اتصال
# ----------------------------------------------------------------------
def test_connection() -> bool:
    base_url = DATABASE_URL.rsplit('/', 1)[0]
    target_url = f"{base_url}/{DB_NAME}"
    try:
        conn = psycopg2.connect(target_url)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        cur.close()
        conn.close()
        logger.info("اتصال به دیتابیس تست شد.")
        return True
    except Exception as e:
        logger.error(f"اتصال ناموفق: {e}")
        return False

# ----------------------------------------------------------------------
# اجرای اصلی
# ----------------------------------------------------------------------
def main():
    logger.info("شروع راه‌اندازی دیتابیس امن پریمیوم ملت...")
    
    if not create_database_safely():
        sys.exit(1)
    
    if not run_sql_script():
        sys.exit(1)
    
    if not test_connection():
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("دیتابیس mellat_premium با موفقیت و به صورت کاملاً امن راه‌اندازی شد.")
    logger.info(f"دیتابیس: {DB_NAME}")
    logger.info(f"فایل SQL: {SQL_FILE}")
    logger.info("=" * 70)
    logger.info("برای به‌روزرسانی روزانه:")
    logger.info(f"  psql -d {DB_NAME} -c \"SELECT refresh_daily_summary();\"")
    logger.info("=" * 70)

# ----------------------------------------------------------------------
# اجرا
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()