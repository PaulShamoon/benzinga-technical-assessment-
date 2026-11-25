WITH daily AS (
    SELECT
        model_name,
        created_at::date AS day,
        COUNT(*) AS n,
        SUM(CASE WHEN predicted = actual THEN 1 ELSE 0 END) AS correct_n,
        100.0 * SUM(CASE WHEN predicted = actual THEN 1 ELSE 0 END) / COUNT(*) AS daily_acc,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency
    FROM inferences
    GROUP BY model_name, created_at::date
),

windowed AS (
    SELECT
        model_name,
        day,
        n,
        daily_acc,
        p95_latency,
        100.0 * SUM(correct_n) OVER seven_days / SUM(n) OVER seven_days AS rolling_acc_7d,
        daily_acc - LAG(daily_acc) OVER (PARTITION BY model_name ORDER BY day) AS acc_delta
    FROM daily
    WINDOW seven_days AS (
        PARTITION BY model_name
        ORDER BY day
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
    )
)

SELECT
    model_name,
    day,
    n,
    ROUND(daily_acc, 2) AS daily_acc,
    ROUND(rolling_acc_7d, 2) AS rolling_acc_7d,
    ROUND(acc_delta, 2) AS acc_delta,
    p95_latency,
    CASE WHEN acc_delta < -10 THEN 1 ELSE 0 END AS drift_flag
FROM windowed
WHERE n >= 100
ORDER BY model_name, day;
