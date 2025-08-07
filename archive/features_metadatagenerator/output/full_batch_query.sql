WITH raw AS (
 SELECT 
    uid as bnc_uid
    , DATE('${%Y-%m-%d,-0}') AS partition_date
    , create_time as login_unix_ts
    , unix_timestamp(
      to_utc_timestamp(
            cast(DATE('${%Y-%m-%d,-0}') as timestamp),
            'Asia/Jakarta'
        )
      ) * 1000 AS partition_date_unix_ts
    , id as login_id
  FROM hive.snap.mob_t_login_device_log
  WHERE TRUE 
  AND success_flag = 0
  AND deleted = 0
  AND create_date = '20250805'
  AND app_login = 1
  AND create_date >= date_format(date_sub(DATE('${%Y-%m-%d,-0}'), 90), 'yyyyMMdd') -- start at 90 days prior
  AND create_date < date_format(DATE('${%Y-%m-%d,-0}'), 'yyyyMMdd') -- end at midnight today
  )
  , prep AS (
    SELECT 
      bnc_uid
      , partition_date
      , CASE 
        WHEN device_type = 'ios' THEN TRUE 
        ELSE FALSE 
        END as is_ios
      , login_id 
      , login_unix_ts
      , (login_unix_ts BETWEEN partition_date_unix_ts - 86400000 AND partition_date_unix_ts) as l1d
      , (login_unix_ts BETWEEN partition_date_unix_ts - 259200000 AND partition_date_unix_ts) as l3d
      , (login_unix_ts BETWEEN partition_date_unix_ts - 604800000 AND partition_date_unix_ts) as l7d
      , (login_unix_ts BETWEEN partition_date_unix_ts - 1209600000 AND partition_date_unix_ts) as l14d
      , (login_unix_ts BETWEEN partition_date_unix_ts - 2592000000 AND partition_date_unix_ts) as l30d
      , (login_unix_ts BETWEEN partition_date_unix_ts - 5184000000 AND partition_date_unix_ts) as l60d
      , (login_unix_ts BETWEEN partition_date_unix_ts - 7776000000 AND partition_date_unix_ts) as l90d
      , (login_unix_ts - 631152000000) / (partition_date_unix_ts - 631152000000) AS scaled_diff_login_unix_ts
    FROM raw
  )
, feat_pc_login_hist_l90d AS (
  SELECT
    bnc_uid
    , COUNT(CASE WHEN l1d THEN login_id ELSE NULL END) AS count_login_id_l1d
    , COUNT(CASE WHEN l3d THEN login_id ELSE NULL END) AS count_login_id_l3d
    , COUNT(CASE WHEN l7d THEN login_id ELSE NULL END) AS count_login_id_l7d
    , COUNT(CASE WHEN l14d THEN login_id ELSE NULL END) AS count_login_id_l14d
    , COUNT(CASE WHEN l30d THEN login_id ELSE NULL END) AS count_login_id_l30d
    , COUNT(CASE WHEN l60d THEN login_id ELSE NULL END) AS count_login_id_l60d
    , COUNT(CASE WHEN l90d THEN login_id ELSE NULL END) AS count_login_id_l90d
    , COUNT(CASE WHEN is_ios AND l1d THEN login_id ELSE NULL END) AS count_is_ios_login_id_l1d
    , COUNT(CASE WHEN is_ios AND l3d THEN login_id ELSE NULL END) AS count_is_ios_login_id_l3d
    , COUNT(CASE WHEN is_ios AND l7d THEN login_id ELSE NULL END) AS count_is_ios_login_id_l7d
    , COUNT(CASE WHEN is_ios AND l14d THEN login_id ELSE NULL END) AS count_is_ios_login_id_l14d
    , COUNT(CASE WHEN is_ios AND l30d THEN login_id ELSE NULL END) AS count_is_ios_login_id_l30d
    , COUNT(CASE WHEN is_ios AND l60d THEN login_id ELSE NULL END) AS count_is_ios_login_id_l60d
    , COUNT(CASE WHEN is_ios AND l90d THEN login_id ELSE NULL END) AS count_is_ios_login_id_l90d
    , COUNT(CASE WHEN (NOT is_ios) AND l1d THEN login_id ELSE NULL END) AS count_is_notios_login_id_l1d
    , COUNT(CASE WHEN (NOT is_ios) AND l3d THEN login_id ELSE NULL END) AS count_is_notios_login_id_l3d
    , COUNT(CASE WHEN (NOT is_ios) AND l7d THEN login_id ELSE NULL END) AS count_is_notios_login_id_l7d
    , COUNT(CASE WHEN (NOT is_ios) AND l14d THEN login_id ELSE NULL END) AS count_is_notios_login_id_l14d
    , COUNT(CASE WHEN (NOT is_ios) AND l30d THEN login_id ELSE NULL END) AS count_is_notios_login_id_l30d
    , COUNT(CASE WHEN (NOT is_ios) AND l60d THEN login_id ELSE NULL END) AS count_is_notios_login_id_l60d
    , COUNT(CASE WHEN (NOT is_ios) AND l90d THEN login_id ELSE NULL END) AS count_is_notios_login_id_l90d
    , MIN(CASE WHEN l1d THEN is_ios ELSE NULL END) AS min_is_ios_l1d
    , MIN(CASE WHEN l3d THEN is_ios ELSE NULL END) AS min_is_ios_l3d
    , MIN(CASE WHEN l7d THEN is_ios ELSE NULL END) AS min_is_ios_l7d
    , MIN(CASE WHEN l14d THEN is_ios ELSE NULL END) AS min_is_ios_l14d
    , MIN(CASE WHEN l30d THEN is_ios ELSE NULL END) AS min_is_ios_l30d
    , MIN(CASE WHEN l60d THEN is_ios ELSE NULL END) AS min_is_ios_l60d
    , MIN(CASE WHEN l90d THEN is_ios ELSE NULL END) AS min_is_ios_l90d
    , MAX(CASE WHEN l1d THEN is_ios ELSE NULL END) AS max_is_ios_l1d
    , MAX(CASE WHEN l3d THEN is_ios ELSE NULL END) AS max_is_ios_l3d
    , MAX(CASE WHEN l7d THEN is_ios ELSE NULL END) AS max_is_ios_l7d
    , MAX(CASE WHEN l14d THEN is_ios ELSE NULL END) AS max_is_ios_l14d
    , MAX(CASE WHEN l30d THEN is_ios ELSE NULL END) AS max_is_ios_l30d
    , MAX(CASE WHEN l60d THEN is_ios ELSE NULL END) AS max_is_ios_l60d
    , MAX(CASE WHEN l90d THEN is_ios ELSE NULL END) AS max_is_ios_l90d
    , AVG(CASE WHEN l1d THEN is_ios ELSE NULL END) AS avg_is_ios_l1d
    , AVG(CASE WHEN l3d THEN is_ios ELSE NULL END) AS avg_is_ios_l3d
    , AVG(CASE WHEN l7d THEN is_ios ELSE NULL END) AS avg_is_ios_l7d
    , AVG(CASE WHEN l14d THEN is_ios ELSE NULL END) AS avg_is_ios_l14d
    , AVG(CASE WHEN l30d THEN is_ios ELSE NULL END) AS avg_is_ios_l30d
    , AVG(CASE WHEN l60d THEN is_ios ELSE NULL END) AS avg_is_ios_l60d
    , AVG(CASE WHEN l90d THEN is_ios ELSE NULL END) AS avg_is_ios_l90d
    , MIN(CASE WHEN is_ios AND l1d THEN is_ios ELSE NULL END) AS min_is_ios_is_ios_l1d
    , MIN(CASE WHEN is_ios AND l3d THEN is_ios ELSE NULL END) AS min_is_ios_is_ios_l3d
    , MIN(CASE WHEN is_ios AND l7d THEN is_ios ELSE NULL END) AS min_is_ios_is_ios_l7d
    , MIN(CASE WHEN is_ios AND l14d THEN is_ios ELSE NULL END) AS min_is_ios_is_ios_l14d
    , MIN(CASE WHEN is_ios AND l30d THEN is_ios ELSE NULL END) AS min_is_ios_is_ios_l30d
    , MIN(CASE WHEN is_ios AND l60d THEN is_ios ELSE NULL END) AS min_is_ios_is_ios_l60d
    , MIN(CASE WHEN is_ios AND l90d THEN is_ios ELSE NULL END) AS min_is_ios_is_ios_l90d
    , MAX(CASE WHEN is_ios AND l1d THEN is_ios ELSE NULL END) AS max_is_ios_is_ios_l1d
    , MAX(CASE WHEN is_ios AND l3d THEN is_ios ELSE NULL END) AS max_is_ios_is_ios_l3d
    , MAX(CASE WHEN is_ios AND l7d THEN is_ios ELSE NULL END) AS max_is_ios_is_ios_l7d
    , MAX(CASE WHEN is_ios AND l14d THEN is_ios ELSE NULL END) AS max_is_ios_is_ios_l14d
    , MAX(CASE WHEN is_ios AND l30d THEN is_ios ELSE NULL END) AS max_is_ios_is_ios_l30d
    , MAX(CASE WHEN is_ios AND l60d THEN is_ios ELSE NULL END) AS max_is_ios_is_ios_l60d
    , MAX(CASE WHEN is_ios AND l90d THEN is_ios ELSE NULL END) AS max_is_ios_is_ios_l90d
    , AVG(CASE WHEN is_ios AND l1d THEN is_ios ELSE NULL END) AS avg_is_ios_is_ios_l1d
    , AVG(CASE WHEN is_ios AND l3d THEN is_ios ELSE NULL END) AS avg_is_ios_is_ios_l3d
    , AVG(CASE WHEN is_ios AND l7d THEN is_ios ELSE NULL END) AS avg_is_ios_is_ios_l7d
    , AVG(CASE WHEN is_ios AND l14d THEN is_ios ELSE NULL END) AS avg_is_ios_is_ios_l14d
    , AVG(CASE WHEN is_ios AND l30d THEN is_ios ELSE NULL END) AS avg_is_ios_is_ios_l30d
    , AVG(CASE WHEN is_ios AND l60d THEN is_ios ELSE NULL END) AS avg_is_ios_is_ios_l60d
    , AVG(CASE WHEN is_ios AND l90d THEN is_ios ELSE NULL END) AS avg_is_ios_is_ios_l90d
    , MIN(CASE WHEN (NOT is_ios) AND l1d THEN is_ios ELSE NULL END) AS min_is_notios_is_ios_l1d
    , MIN(CASE WHEN (NOT is_ios) AND l3d THEN is_ios ELSE NULL END) AS min_is_notios_is_ios_l3d
    , MIN(CASE WHEN (NOT is_ios) AND l7d THEN is_ios ELSE NULL END) AS min_is_notios_is_ios_l7d
    , MIN(CASE WHEN (NOT is_ios) AND l14d THEN is_ios ELSE NULL END) AS min_is_notios_is_ios_l14d
    , MIN(CASE WHEN (NOT is_ios) AND l30d THEN is_ios ELSE NULL END) AS min_is_notios_is_ios_l30d
    , MIN(CASE WHEN (NOT is_ios) AND l60d THEN is_ios ELSE NULL END) AS min_is_notios_is_ios_l60d
    , MIN(CASE WHEN (NOT is_ios) AND l90d THEN is_ios ELSE NULL END) AS min_is_notios_is_ios_l90d
    , MAX(CASE WHEN (NOT is_ios) AND l1d THEN is_ios ELSE NULL END) AS max_is_notios_is_ios_l1d
    , MAX(CASE WHEN (NOT is_ios) AND l3d THEN is_ios ELSE NULL END) AS max_is_notios_is_ios_l3d
    , MAX(CASE WHEN (NOT is_ios) AND l7d THEN is_ios ELSE NULL END) AS max_is_notios_is_ios_l7d
    , MAX(CASE WHEN (NOT is_ios) AND l14d THEN is_ios ELSE NULL END) AS max_is_notios_is_ios_l14d
    , MAX(CASE WHEN (NOT is_ios) AND l30d THEN is_ios ELSE NULL END) AS max_is_notios_is_ios_l30d
    , MAX(CASE WHEN (NOT is_ios) AND l60d THEN is_ios ELSE NULL END) AS max_is_notios_is_ios_l60d
    , MAX(CASE WHEN (NOT is_ios) AND l90d THEN is_ios ELSE NULL END) AS max_is_notios_is_ios_l90d
    , AVG(CASE WHEN (NOT is_ios) AND l1d THEN is_ios ELSE NULL END) AS avg_is_notios_is_ios_l1d
    , AVG(CASE WHEN (NOT is_ios) AND l3d THEN is_ios ELSE NULL END) AS avg_is_notios_is_ios_l3d
    , AVG(CASE WHEN (NOT is_ios) AND l7d THEN is_ios ELSE NULL END) AS avg_is_notios_is_ios_l7d
    , AVG(CASE WHEN (NOT is_ios) AND l14d THEN is_ios ELSE NULL END) AS avg_is_notios_is_ios_l14d
    , AVG(CASE WHEN (NOT is_ios) AND l30d THEN is_ios ELSE NULL END) AS avg_is_notios_is_ios_l30d
    , AVG(CASE WHEN (NOT is_ios) AND l60d THEN is_ios ELSE NULL END) AS avg_is_notios_is_ios_l60d
    , AVG(CASE WHEN (NOT is_ios) AND l90d THEN is_ios ELSE NULL END) AS avg_is_notios_is_ios_l90d
    , MIN(CASE WHEN l1d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_scaled_diff_login_unix_ts_l1d
    , MIN(CASE WHEN l3d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_scaled_diff_login_unix_ts_l3d
    , MIN(CASE WHEN l7d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_scaled_diff_login_unix_ts_l7d
    , MIN(CASE WHEN l14d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_scaled_diff_login_unix_ts_l14d
    , MIN(CASE WHEN l30d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_scaled_diff_login_unix_ts_l30d
    , MIN(CASE WHEN l60d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_scaled_diff_login_unix_ts_l60d
    , MIN(CASE WHEN l90d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_scaled_diff_login_unix_ts_l90d
    , MAX(CASE WHEN l1d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_scaled_diff_login_unix_ts_l1d
    , MAX(CASE WHEN l3d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_scaled_diff_login_unix_ts_l3d
    , MAX(CASE WHEN l7d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_scaled_diff_login_unix_ts_l7d
    , MAX(CASE WHEN l14d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_scaled_diff_login_unix_ts_l14d
    , MAX(CASE WHEN l30d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_scaled_diff_login_unix_ts_l30d
    , MAX(CASE WHEN l60d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_scaled_diff_login_unix_ts_l60d
    , MAX(CASE WHEN l90d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_scaled_diff_login_unix_ts_l90d
    , AVG(CASE WHEN l1d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_scaled_diff_login_unix_ts_l1d
    , AVG(CASE WHEN l3d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_scaled_diff_login_unix_ts_l3d
    , AVG(CASE WHEN l7d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_scaled_diff_login_unix_ts_l7d
    , AVG(CASE WHEN l14d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_scaled_diff_login_unix_ts_l14d
    , AVG(CASE WHEN l30d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_scaled_diff_login_unix_ts_l30d
    , AVG(CASE WHEN l60d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_scaled_diff_login_unix_ts_l60d
    , AVG(CASE WHEN l90d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_scaled_diff_login_unix_ts_l90d
    , MIN(CASE WHEN is_ios AND l1d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_ios_scaled_diff_login_unix_ts_l1d
    , MIN(CASE WHEN is_ios AND l3d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_ios_scaled_diff_login_unix_ts_l3d
    , MIN(CASE WHEN is_ios AND l7d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_ios_scaled_diff_login_unix_ts_l7d
    , MIN(CASE WHEN is_ios AND l14d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_ios_scaled_diff_login_unix_ts_l14d
    , MIN(CASE WHEN is_ios AND l30d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_ios_scaled_diff_login_unix_ts_l30d
    , MIN(CASE WHEN is_ios AND l60d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_ios_scaled_diff_login_unix_ts_l60d
    , MIN(CASE WHEN is_ios AND l90d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_ios_scaled_diff_login_unix_ts_l90d
    , MAX(CASE WHEN is_ios AND l1d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_ios_scaled_diff_login_unix_ts_l1d
    , MAX(CASE WHEN is_ios AND l3d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_ios_scaled_diff_login_unix_ts_l3d
    , MAX(CASE WHEN is_ios AND l7d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_ios_scaled_diff_login_unix_ts_l7d
    , MAX(CASE WHEN is_ios AND l14d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_ios_scaled_diff_login_unix_ts_l14d
    , MAX(CASE WHEN is_ios AND l30d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_ios_scaled_diff_login_unix_ts_l30d
    , MAX(CASE WHEN is_ios AND l60d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_ios_scaled_diff_login_unix_ts_l60d
    , MAX(CASE WHEN is_ios AND l90d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_ios_scaled_diff_login_unix_ts_l90d
    , AVG(CASE WHEN is_ios AND l1d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_ios_scaled_diff_login_unix_ts_l1d
    , AVG(CASE WHEN is_ios AND l3d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_ios_scaled_diff_login_unix_ts_l3d
    , AVG(CASE WHEN is_ios AND l7d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_ios_scaled_diff_login_unix_ts_l7d
    , AVG(CASE WHEN is_ios AND l14d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_ios_scaled_diff_login_unix_ts_l14d
    , AVG(CASE WHEN is_ios AND l30d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_ios_scaled_diff_login_unix_ts_l30d
    , AVG(CASE WHEN is_ios AND l60d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_ios_scaled_diff_login_unix_ts_l60d
    , AVG(CASE WHEN is_ios AND l90d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_ios_scaled_diff_login_unix_ts_l90d
    , MIN(CASE WHEN (NOT is_ios) AND l1d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_notios_scaled_diff_login_unix_ts_l1d
    , MIN(CASE WHEN (NOT is_ios) AND l3d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_notios_scaled_diff_login_unix_ts_l3d
    , MIN(CASE WHEN (NOT is_ios) AND l7d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_notios_scaled_diff_login_unix_ts_l7d
    , MIN(CASE WHEN (NOT is_ios) AND l14d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_notios_scaled_diff_login_unix_ts_l14d
    , MIN(CASE WHEN (NOT is_ios) AND l30d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_notios_scaled_diff_login_unix_ts_l30d
    , MIN(CASE WHEN (NOT is_ios) AND l60d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_notios_scaled_diff_login_unix_ts_l60d
    , MIN(CASE WHEN (NOT is_ios) AND l90d THEN scaled_diff_login_unix_ts ELSE NULL END) AS min_is_notios_scaled_diff_login_unix_ts_l90d
    , MAX(CASE WHEN (NOT is_ios) AND l1d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_notios_scaled_diff_login_unix_ts_l1d
    , MAX(CASE WHEN (NOT is_ios) AND l3d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_notios_scaled_diff_login_unix_ts_l3d
    , MAX(CASE WHEN (NOT is_ios) AND l7d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_notios_scaled_diff_login_unix_ts_l7d
    , MAX(CASE WHEN (NOT is_ios) AND l14d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_notios_scaled_diff_login_unix_ts_l14d
    , MAX(CASE WHEN (NOT is_ios) AND l30d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_notios_scaled_diff_login_unix_ts_l30d
    , MAX(CASE WHEN (NOT is_ios) AND l60d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_notios_scaled_diff_login_unix_ts_l60d
    , MAX(CASE WHEN (NOT is_ios) AND l90d THEN scaled_diff_login_unix_ts ELSE NULL END) AS max_is_notios_scaled_diff_login_unix_ts_l90d
    , AVG(CASE WHEN (NOT is_ios) AND l1d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_notios_scaled_diff_login_unix_ts_l1d
    , AVG(CASE WHEN (NOT is_ios) AND l3d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_notios_scaled_diff_login_unix_ts_l3d
    , AVG(CASE WHEN (NOT is_ios) AND l7d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_notios_scaled_diff_login_unix_ts_l7d
    , AVG(CASE WHEN (NOT is_ios) AND l14d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_notios_scaled_diff_login_unix_ts_l14d
    , AVG(CASE WHEN (NOT is_ios) AND l30d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_notios_scaled_diff_login_unix_ts_l30d
    , AVG(CASE WHEN (NOT is_ios) AND l60d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_notios_scaled_diff_login_unix_ts_l60d
    , AVG(CASE WHEN (NOT is_ios) AND l90d THEN scaled_diff_login_unix_ts ELSE NULL END) AS avg_is_notios_scaled_diff_login_unix_ts_l90d
    , MIN(CASE WHEN l1d THEN login_unix_ts ELSE NULL END) AS min_login_unix_ts_l1d
    , MIN(CASE WHEN l3d THEN login_unix_ts ELSE NULL END) AS min_login_unix_ts_l3d
    , MIN(CASE WHEN l7d THEN login_unix_ts ELSE NULL END) AS min_login_unix_ts_l7d
    , MIN(CASE WHEN l14d THEN login_unix_ts ELSE NULL END) AS min_login_unix_ts_l14d
    , MIN(CASE WHEN l30d THEN login_unix_ts ELSE NULL END) AS min_login_unix_ts_l30d
    , MIN(CASE WHEN l60d THEN login_unix_ts ELSE NULL END) AS min_login_unix_ts_l60d
    , MIN(CASE WHEN l90d THEN login_unix_ts ELSE NULL END) AS min_login_unix_ts_l90d
    , MAX(CASE WHEN l1d THEN login_unix_ts ELSE NULL END) AS max_login_unix_ts_l1d
    , MAX(CASE WHEN l3d THEN login_unix_ts ELSE NULL END) AS max_login_unix_ts_l3d
    , MAX(CASE WHEN l7d THEN login_unix_ts ELSE NULL END) AS max_login_unix_ts_l7d
    , MAX(CASE WHEN l14d THEN login_unix_ts ELSE NULL END) AS max_login_unix_ts_l14d
    , MAX(CASE WHEN l30d THEN login_unix_ts ELSE NULL END) AS max_login_unix_ts_l30d
    , MAX(CASE WHEN l60d THEN login_unix_ts ELSE NULL END) AS max_login_unix_ts_l60d
    , MAX(CASE WHEN l90d THEN login_unix_ts ELSE NULL END) AS max_login_unix_ts_l90d
    , MIN(CASE WHEN is_ios AND l1d THEN login_unix_ts ELSE NULL END) AS min_is_ios_login_unix_ts_l1d
    , MIN(CASE WHEN is_ios AND l3d THEN login_unix_ts ELSE NULL END) AS min_is_ios_login_unix_ts_l3d
    , MIN(CASE WHEN is_ios AND l7d THEN login_unix_ts ELSE NULL END) AS min_is_ios_login_unix_ts_l7d
    , MIN(CASE WHEN is_ios AND l14d THEN login_unix_ts ELSE NULL END) AS min_is_ios_login_unix_ts_l14d
    , MIN(CASE WHEN is_ios AND l30d THEN login_unix_ts ELSE NULL END) AS min_is_ios_login_unix_ts_l30d
    , MIN(CASE WHEN is_ios AND l60d THEN login_unix_ts ELSE NULL END) AS min_is_ios_login_unix_ts_l60d
    , MIN(CASE WHEN is_ios AND l90d THEN login_unix_ts ELSE NULL END) AS min_is_ios_login_unix_ts_l90d
    , MAX(CASE WHEN is_ios AND l1d THEN login_unix_ts ELSE NULL END) AS max_is_ios_login_unix_ts_l1d
    , MAX(CASE WHEN is_ios AND l3d THEN login_unix_ts ELSE NULL END) AS max_is_ios_login_unix_ts_l3d
    , MAX(CASE WHEN is_ios AND l7d THEN login_unix_ts ELSE NULL END) AS max_is_ios_login_unix_ts_l7d
    , MAX(CASE WHEN is_ios AND l14d THEN login_unix_ts ELSE NULL END) AS max_is_ios_login_unix_ts_l14d
    , MAX(CASE WHEN is_ios AND l30d THEN login_unix_ts ELSE NULL END) AS max_is_ios_login_unix_ts_l30d
    , MAX(CASE WHEN is_ios AND l60d THEN login_unix_ts ELSE NULL END) AS max_is_ios_login_unix_ts_l60d
    , MAX(CASE WHEN is_ios AND l90d THEN login_unix_ts ELSE NULL END) AS max_is_ios_login_unix_ts_l90d
    , MIN(CASE WHEN (NOT is_ios) AND l1d THEN login_unix_ts ELSE NULL END) AS min_is_notios_login_unix_ts_l1d
    , MIN(CASE WHEN (NOT is_ios) AND l3d THEN login_unix_ts ELSE NULL END) AS min_is_notios_login_unix_ts_l3d
    , MIN(CASE WHEN (NOT is_ios) AND l7d THEN login_unix_ts ELSE NULL END) AS min_is_notios_login_unix_ts_l7d
    , MIN(CASE WHEN (NOT is_ios) AND l14d THEN login_unix_ts ELSE NULL END) AS min_is_notios_login_unix_ts_l14d
    , MIN(CASE WHEN (NOT is_ios) AND l30d THEN login_unix_ts ELSE NULL END) AS min_is_notios_login_unix_ts_l30d
    , MIN(CASE WHEN (NOT is_ios) AND l60d THEN login_unix_ts ELSE NULL END) AS min_is_notios_login_unix_ts_l60d
    , MIN(CASE WHEN (NOT is_ios) AND l90d THEN login_unix_ts ELSE NULL END) AS min_is_notios_login_unix_ts_l90d
    , MAX(CASE WHEN (NOT is_ios) AND l1d THEN login_unix_ts ELSE NULL END) AS max_is_notios_login_unix_ts_l1d
    , MAX(CASE WHEN (NOT is_ios) AND l3d THEN login_unix_ts ELSE NULL END) AS max_is_notios_login_unix_ts_l3d
    , MAX(CASE WHEN (NOT is_ios) AND l7d THEN login_unix_ts ELSE NULL END) AS max_is_notios_login_unix_ts_l7d
    , MAX(CASE WHEN (NOT is_ios) AND l14d THEN login_unix_ts ELSE NULL END) AS max_is_notios_login_unix_ts_l14d
    , MAX(CASE WHEN (NOT is_ios) AND l30d THEN login_unix_ts ELSE NULL END) AS max_is_notios_login_unix_ts_l30d
    , MAX(CASE WHEN (NOT is_ios) AND l60d THEN login_unix_ts ELSE NULL END) AS max_is_notios_login_unix_ts_l60d
    , MAX(CASE WHEN (NOT is_ios) AND l90d THEN login_unix_ts ELSE NULL END) AS max_is_notios_login_unix_ts_l90d
    , partition_date
  FROM prep
  GROUP BY bnc_uid
         , partition_date
)
SELECT 
  *
  , DATE('${%Y-%m-%d,-0}') AS etl_date
FROM feat_pc_login_hist_l90d
DISTRIBUTE BY partition_date;