# mellat_premium_analytics_pro.py
# نسخه حرفه‌ای، یکپارچه، کارآمد و تولید-محور
# تحلیل داده + آموزش مودلی + پیش‌بینی درآمد + تشخیص تقلب
# اجرا: python mellat_premium_analytics_pro.py

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
from sqlalchemy import create_engine, text
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# تنظیمات کلی
# ----------------------------------------------------------------------
@dataclass
class Config:
    DATA_DIR: Path = Path("analytics_data")
    MODEL_DIR: Path = Path("models")
    LOG_LEVEL: int = logging.INFO
    MOCK_DATA_SIZE: int = 50000
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    FONT_FAMILY: str = 'Tahoma'
    FIGSIZE_DAILY: tuple = (14, 7)
    FIGSIZE_HOURLY: tuple = (12, 6)
    FORECAST_DAYS: int = 7
    DATABASE_URL: Optional[str] = os.getenv(
        "DATABASE_URL",
        "postgresql://mellat:securepassword123@db:5432/mellat"
    )

config = Config()
config.DATA_DIR.mkdir(exist_ok=True)
config.MODEL_DIR.mkdir(exist_ok=True)

# تنظیمات لاگ و نمودار
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mellat_analytics_pro")
sns.set_style("whitegrid")
plt.rcParams['font.family'] = config.FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------------------
# اتصال به دیتابیس
# ----------------------------------------------------------------------
def get_db_engine():
    try:
        engine = create_engine(config.DATABASE_URL, pool_pre_ping=True, echo=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("اتصال به دیتابیس PostgreSQL برقرار شد.")
        return engine
    except Exception as e:
        logger.warning(f"اتصال به دیتابیس ناموفق: {e}. استفاده از داده‌های شبیه‌سازی.")
        return None

# ----------------------------------------------------------------------
# بارگذاری داده از دیتابیس یا شبیه‌سازی
# ----------------------------------------------------------------------
def load_data_from_db(engine) -> pd.DataFrame:
    query = """
    SELECT 
        p.order_id, p.amount, p.status, p.created_at, p.metadata
    FROM payments p
    WHERE p.created_at >= NOW() - INTERVAL '90 days'
    """
    try:
        df = pd.read_sql(query, engine, parse_dates=["created_at"])
        # استخراج user_id و device_id از metadata
        if 'metadata' in df.columns and not df['metadata'].isna().all():
            meta = df['metadata'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            df['user_id'] = meta.apply(lambda x: x.get('user_id') if isinstance(x, dict) else None)
            df['device_id'] = meta.apply(lambda x: x.get('device_id') if isinstance(x, dict) else None)
        else:
            df['user_id'] = df['device_id'] = None
        logger.info(f"داده‌ها از دیتابیس بارگذاری شد: {len(df)} رکورد")
        return df
    except Exception as e:
        logger.error(f"خطا در بارگذاری از دیتابیس: {e}")
        return pd.DataFrame()

def generate_mock_data() -> pd.DataFrame:
    n = config.MOCK_DATA_SIZE
    np.random.seed(config.RANDOM_STATE)

    users = [f"user_{i}" for i in range(1, min(n//10 + 1, 2000))]
    devices = [f"dev_{i}" for i in range(1, min(n//5 + 1, 5000))]
    start = datetime(2024, 1, 1)
    times = [start + timedelta(minutes=30 * i) for i in range(n)]

    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "user_id": np.random.choice(users, n),
        "device_id": np.random.choice(devices, n),
        "amount": 300000,
        "status": np.random.choice(["success", "failed", "pending"], n, p=[0.93, 0.05, 0.02]),
        "created_at": np.random.choice(times, n),
        "ip_address": [".".join(map(str, np.random.randint(0, 256, 4))) for _ in range(n)],
        "user_agent": np.random.choice([
            "Android/Chrome", "iOS/Safari", "Windows/Chrome", "Mac/Safari", "Linux/Firefox"
        ], n),
    })

    df["hour"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].between(0, 6).astype(int)
    df["payment_success"] = (df["status"] == "success").astype(int)

    # شبیه‌سازی تقلب
    fraud_mask = (
        (df["is_night"] == 1) |
        (df["user_agent"] == "Linux/Firefox") |
        (df["ip_address"].str.startswith(("192.168", "10.", "172.16")))
    )
    df.loc[fraud_mask & (df["status"] == "success"), "status"] = "failed"
    df.loc[fraud_mask, "payment_success"] = 0

    path = config.DATA_DIR / "payments_raw.csv"
    df.to_csv(path, index=False)
    logger.info(f"داده‌های شبیه‌سازی ({n:,} رکورد) ذخیره شد: {path}")
    return df

def load_data() -> pd.DataFrame:
    engine = get_db_engine()
    if engine:
        df_db = load_data_from_db(engine)
        if not df_db.empty and 'user_id' in df_db.columns:
            return df_db
    return generate_mock_data()

# ----------------------------------------------------------------------
# تحلیل اکتشافی (EDA)
# ----------------------------------------------------------------------
def eda_analysis(df: pd.DataFrame):
    logger.info("شروع تحلیل اکتشافی (EDA)...")

    total_revenue = df[df["status"] == "success"]["amount"].sum()
    success_rate = (df["status"] == "success").mean() * 100
    active_users = df[df["status"] == "success"]["user_id"].nunique() if 'user_id' in df.columns else 0
    total_transactions = len(df)

    print(f"\n{'='*60}")
    print(f"تحلیل کلی درگاه پریمیوم")
    print(f"تعداد تراکنش: {total_transactions:,}")
    print(f"درآمد کل: {total_revenue:,} تومان")
    print(f"نرخ موفقیت: {success_rate:.2f}%")
    print(f"کاربران فعال: {active_users:,}")
    print(f"{'='*60}\n")

    # نمودار درآمد روزانه
    daily = df[df["status"] == "success"].groupby(df["created_at"].dt.date)["amount"].sum()
    plt.figure(figsize=config.FIGSIZE_DAILY)
    daily.plot(marker="o", color="#10b981", linewidth=2)
    plt.title("روند درآمد روزانه (تومان)", fontsize=16, fontweight="bold")
    plt.ylabel("درآمد")
    plt.xlabel("تاریخ")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(config.DATA_DIR / "daily_revenue.png", dpi=300, bbox_inches='tight')
    plt.close()

    # نمودار ساعتی
    hourly = df[df["status"] == "success"].groupby("hour").size()
    plt.figure(figsize=config.FIGSIZE_HOURLY)
    hourly.plot(kind="bar", color="#3b82f6")
    plt.title("پرداخت‌های موفق بر اساس ساعت", fontsize=16, fontweight="bold")
    plt.xlabel("ساعت")
    plt.ylabel("تعداد")
    plt.tight_layout()
    plt.savefig(config.DATA_DIR / "hourly_payments.png", dpi=300, bbox_inches='tight')
    plt.close()

    # توزیع وضعیت
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="status", palette="viridis", order=["success", "failed", "pending"])
    plt.title("توزیع وضعیت پرداخت‌ها", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.DATA_DIR / "status_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("نمودارهای EDA ذخیره شدند.")

# ----------------------------------------------------------------------
# مهندسی ویژگی
# ----------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info("مهندسی ویژگی...")

    # اطمینان از وجود ستون‌های ضروری
    required = ["user_id", "device_id", "ip_address", "user_agent"]
    for col in required:
        if col not in df.columns:
            df[col] = "unknown"

    # انکودرها
    encoders = {}
    for col in ["user_id", "device_id", "ip_address", "user_agent"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        joblib.dump(le, config.MODEL_DIR / f"encoder_{col}.pkl")

    # ویژگی‌های رفتاری
    df["payments_per_user"] = df.groupby("user_id")["payment_success"].transform("sum")
    df["failed_per_user"] = df.groupby("user_id")["status"].transform(lambda x: (x == "failed").sum())
    df["is_repeat"] = (df["payments_per_user"] > 1).astype(int)
    df["is_suspicious_ip"] = df["ip_address"].str.contains(r"^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)").astype(int)
    df["is_mobile"] = df["user_agent"].str.contains("Android|iOS", case=False).astype(int)

    # برچسب تقلب
    df["is_fraud"] = (
        (df["is_night"] == 1) |
        (df["is_suspicious_ip"] == 1) |
        (df["user_agent"] == "Linux/Firefox")
    ).astype(int)

    path = config.DATA_DIR / "features_engineered.csv"
    df.to_csv(path, index=False)
    logger.info(f"ویژگی‌ها ذخیره شد: {path}")
    return df

# ----------------------------------------------------------------------
# مدل تشخیص تقلب
# ----------------------------------------------------------------------
def train_fraud_model(df: pd.DataFrame):
    logger.info("آموزش مدل تشخیص تقلب...")
    features = [
        "hour", "day_of_week", "is_weekend", "is_night",
        "user_id_enc", "device_id_enc", "ip_address_enc", "user_agent_enc",
        "payments_per_user", "failed_per_user", "is_repeat",
        "is_suspicious_ip", "is_mobile"
    ]
    X = df[features]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=config.RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nگزارش تشخیص تقلب:")
    print(classification_report(y_test, y_pred, target_names=["عادی", "تقلب"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["عادی", "تقلب"], yticklabels=["عادی", "تقلب"])
    plt.title("ماتریس درهم‌ریختگی - تشخیص تقلب")
    plt.ylabel("واقعی")
    plt.xlabel("پیش‌بینی")
    plt.tight_layout()
    plt.savefig(config.DATA_DIR / "fraud_confusion_matrix.png", dpi=300)
    plt.close()

    joblib.dump(model, config.MODEL_DIR / "fraud_detection_model.pkl")
    logger.info("مدل تقلب ذخیره شد.")
    return model

# ----------------------------------------------------------------------
# پیش‌بینی درآمد
# ----------------------------------------------------------------------
def train_revenue_model(df: pd.DataFrame):
    logger.info("آموزش مدل پیش‌بینی درآمد...")
    daily = df[df["status"] == "success"].groupby(df["created_at"].dt.date)["amount"].sum().reset_index()
    daily["created_at"] = pd.to_datetime(daily["created_at"])
    daily["date_ordinal"] = daily["created_at"].dt.toordinal
    daily["day_of_week"] = daily["created_at"].dt.dayofweek
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)

    X = daily[["date_ordinal", "day_of_week", "is_weekend"]]
    y = daily["amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nپیش‌بینی درآمد - MAE: {mean_absolute_error(y_test, y_pred):,.0f} تومان | R²: {r2_score(y_test, y_pred):.3f}")

    # پیش‌بینی آینده
    last_date = daily["created_at"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, config.FORECAST_DAYS + 1)]
    future_df = pd.DataFrame({
        "created_at": future_dates,
        "date_ordinal": [d.toordinal() for d in future_dates],
        "day_of_week": [d.weekday() for d in future_dates],
        "is_weekend": [1 if d.weekday() in [5, 6] else 0 for d in future_dates]
    })

    pred = model.predict(future_df[["date_ordinal", "day_of_week", "is_weekend"]])
    forecast = future_df.copy()
    forecast["predicted_revenue"] = pred.round(0).astype(int)

    plt.figure(figsize=(12, 6))
    plt.plot(daily["created_at"], daily["amount"], label="واقعی", marker="o")
    plt.plot(forecast["created_at"], forecast["predicted_revenue"], label="پیش‌بینی", marker="x", linestyle="--", color="red")
    plt.title(f"پیش‌بینی درآمد {config.FORECAST_DAYS} روز آینده")
    plt.xlabel("تاریخ")
    plt.ylabel("درآمد (تومان)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(config.DATA_DIR / "revenue_forecast.png", dpi=300)
    plt.close()

    forecast[["created_at", "predicted_revenue"]].to_csv(config.DATA_DIR / "revenue_forecast.csv", index=False)
    joblib.dump(model, config.MODEL_DIR / "revenue_prediction_model.pkl")  # اصلاح شده
    logger.info("مدل درآمد و پیش‌بینی ذخیره شد.")
    return model, forecast

# ----------------------------------------------------------------------
# اجرای اصلی
# ----------------------------------------------------------------------
def main():
    logger.info("شروع تحلیل حرفه‌ای درگاه پریمیوم ملت...")
    df = load_data()
    eda_analysis(df)
    df_feat = feature_engineering(df)
    train_fraud_model(df_feat)
    train_revenue_model(df)

    print(f"\n{'='*70}")
    print(f"تحلیل کامل شد!")
    print(f"داده‌ها: {config.DATA_DIR}")
    print(f"مدل‌ها: {config.MODEL_DIR}")
    print(f"{'='*70}")

# ----------------------------------------------------------------------
# اجرا
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()