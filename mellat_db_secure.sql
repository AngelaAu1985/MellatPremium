-- =============================================================
-- mellat_db_secure.sql
-- دیتابیس کامل، امن و یکپارچه پریمیوم ملت
-- نسخه: 1.0.0 | تاریخ: 2025-04-05
-- اجرا: psql -U postgres -d postgres -f mellat_db_secure.sql
-- =============================================================

-- =============================================================
-- 1. تنظیمات اولیه
-- =============================================================
\set ON_ERROR_STOP on
\echo 'شروع راه‌اندازی دیتابیس امن mellat_premium...'

-- تغییر به دیتابیس postgres برای ایجاد دیتابیس
\c postgres

-- =============================================================
-- 2. ایجاد دیتابیس (اگر وجود نداشته باشد)
-- =============================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_database WHERE datname = 'mellat_premium'
    ) THEN
        EXECUTE 'CREATE DATABASE mellat_premium ENCODING ''UTF8'' TEMPLATE template0';
        RAISE NOTICE 'دیتابیس mellat_premium ایجاد شد.';
    ELSE
        RAISE NOTICE 'دیتابیس mellat_premium قبلاً وجود دارد.';
    END IF;
END $$;

-- اتصال به دیتابیس هدف
\c mellat_premium

-- =============================================================
-- 3. فعال‌سازی افزونه‌ها
-- =============================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
\echo 'افزونه uuid-ossp فعال شد.'

-- =============================================================
-- 4. تابع به‌روزرسانی updated_at
-- =============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================================
-- 5. جدول کاربران (users)
-- =============================================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL UNIQUE,
    device_id VARCHAR(100),
    package_name VARCHAR(100) DEFAULT 'com.yourapp.premium',
    is_premium BOOLEAN DEFAULT FALSE,
    premium_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- شاخص‌ها
CREATE INDEX IF NOT EXISTS ix_users_user_device ON users(user_id, device_id);
CREATE INDEX IF NOT EXISTS ix_users_premium_active ON users(is_premium) WHERE is_premium = TRUE;

-- تریگر
DROP TRIGGER IF EXISTS trg_users_updated ON users;
CREATE TRIGGER trg_users_updated
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

COMMENT ON TABLE users IS 'کاربران اپلیکیشن پریمیوم';
COMMENT ON COLUMN users.is_premium IS 'وضعیت پریمیوم کاربر';

-- =============================================================
-- 6. جدول پرداخت‌ها (payments)
-- =============================================================
CREATE TABLE IF NOT EXISTS payments (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL UNIQUE,
    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE SET NULL,
    amount INTEGER NOT NULL CHECK (amount > 0),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'success', 'failed')),
    ref_id VARCHAR(50),
    sale_order_id INTEGER,
    sale_reference_id VARCHAR(50),
    gateway_response JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- شاخص‌های بهینه
CREATE INDEX IF NOT EXISTS ix_payments_status_created ON payments(status, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_payments_user_status ON payments(user_id, status);
CREATE INDEX IF NOT EXISTS ix_payments_ref ON payments(ref_id) WHERE ref_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_payments_sale_ref ON payments(sale_reference_id) WHERE sale_reference_id IS NOT NULL;

-- تریگر
DROP TRIGGER IF EXISTS trg_payments_updated ON payments;
CREATE TRIGGER trg_payments_updated
    BEFORE UPDATE ON payments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

COMMENT ON TABLE payments IS 'تراکنش‌های درگاه ملت';
COMMENT ON COLUMN payments.amount IS 'مبلغ به ریال';

-- =============================================================
-- 7. جدول لاگ دسترسی (access_logs)
-- =============================================================
CREATE TABLE IF NOT EXISTS access_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50),
    device_id VARCHAR(100),
    access_token TEXT,
    ip_address INET,
    user_agent TEXT,
    endpoint VARCHAR(100),
    is_valid BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- شاخص‌ها
CREATE INDEX IF NOT EXISTS ix_logs_user_token ON access_logs(user_id, access_token);
CREATE INDEX IF NOT EXISTS ix_logs_created_desc ON access_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS ix_logs_invalid ON access_logs(is_valid) WHERE is_valid = FALSE;

COMMENT ON TABLE access_logs IS 'لاگ دسترسی به API';

-- =============================================================
-- 8. جدول خلاصه تحلیلی روزانه (analytics_summary)
-- =============================================================
CREATE TABLE IF NOT EXISTS analytics_summary (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    total_revenue BIGINT DEFAULT 0,
    total_payments INTEGER DEFAULT 0,
    successful_payments INTEGER DEFAULT 0,
    failed_payments INTEGER DEFAULT 0,
    active_users INTEGER DEFAULT 0,
    new_users INTEGER DEFAULT 0,
    fraud_attempts INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_summary_date_desc ON analytics_summary(date DESC);

-- تریگر
DROP TRIGGER IF EXISTS trg_summary_updated ON analytics_summary;
CREATE TRIGGER trg_summary_updated
    BEFORE UPDATE ON analytics_summary
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

COMMENT ON TABLE analytics_summary IS 'خلاصه تحلیلی روزانه';

-- =============================================================
-- 9. ویو: خلاصه 30 روز اخیر
-- =============================================================
CREATE OR REPLACE VIEW v_recent_summary AS
SELECT
    date,
    total_revenue,
    successful_payments,
    failed_payments,
    active_users,
    ROUND(
        CASE WHEN total_payments > 0
        THEN (successful_payments::FLOAT / total_payments) * 100
        ELSE 0 END, 2
    ) AS success_rate_percent
FROM analytics_summary
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY date DESC;

COMMENT ON VIEW v_recent_summary IS 'خلاصه 30 روز اخیر با نرخ موفقیت';

-- =============================================================
-- 10. ویو: کاربران پریمیوم فعال
-- =============================================================
CREATE OR REPLACE VIEW v_active_premium_users AS
SELECT
    user_id,
    device_id,
    package_name,
    premium_expires_at,
    EXTRACT(DAY FROM (premium_expires_at - NOW())) AS days_remaining
FROM users
WHERE is_premium = TRUE
  AND (premium_expires_at IS NULL OR premium_expires_at > NOW());

COMMENT ON VIEW v_active_premium_users IS 'کاربران پریمیوم فعال با روزهای باقی‌مانده';

-- =============================================================
-- 11. ویو: درآمد ساعتی امروز
-- =============================================================
CREATE OR REPLACE VIEW v_hourly_revenue_today AS
SELECT
    EXTRACT(HOUR FROM created_at)::INTEGER AS hour,
    COUNT(*) AS payment_count,
    COALESCE(SUM(CASE WHEN status = 'success' THEN amount ELSE 0 END), 0) AS revenue
FROM payments
WHERE DATE(created_at) = CURRENT_DATE
GROUP BY hour
ORDER BY hour;

COMMENT ON VIEW v_hourly_revenue_today IS 'درآمد ساعتی امروز';

-- =============================================================
-- 12. تابع: به‌روزرسانی خلاصه روزانه (امن)
-- =============================================================
CREATE OR REPLACE FUNCTION refresh_daily_summary(target_date DATE DEFAULT CURRENT_DATE)
RETURNS VOID AS $$
BEGIN
    -- حذف رکورد قدیمی
    DELETE FROM analytics_summary WHERE date = target_date;

    -- درج آمار جدید
    INSERT INTO analytics_summary (
        date, total_revenue, total_payments, successful_payments,
        failed_payments, active_users, new_users
    )
    SELECT
        target_date,
        COALESCE(SUM(CASE WHEN p.status = 'success' THEN p.amount ELSE 0 END), 0),
        COUNT(p.id),
        COUNT(CASE WHEN p.status = 'success' THEN 1 END),
        COUNT(CASE WHEN p.status = 'failed' THEN 1 END),
        COUNT(DISTINCT CASE WHEN u.is_premium THEN u.user_id END),
        COUNT(DISTINCT CASE WHEN DATE(u.created_at) = target_date THEN u.user_id END)
    FROM payments p
    LEFT JOIN users u ON p.user_id = u.user_id
    WHERE DATE(p.created_at) = target_date;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_daily_summary IS 'محاسبه امن خلاصه روزانه - بدون SQLi';

-- =============================================================
-- 13. پایان
-- =============================================================
\echo ''
\echo 'دیتابیس mellat_premium با موفقیت و به صورت امن راه‌اندازی شد.'
\echo 'جداول: users, payments, access_logs, analytics_summary'
\echo 'ویوها: v_recent_summary, v_active_premium_users, v_hourly_revenue_today'
\echo 'تابع: refresh_daily_summary()'
\echo ''
\echo 'برای به‌روزرسانی روزانه:'
\echo '  SELECT refresh_daily_summary();'
\echo ''