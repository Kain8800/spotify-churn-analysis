-- Raw copy of data/spotify_churn_dataset.csv: one row per user, columns in CSV order.
CREATE TABLE raw_users (
    user_id               integer PRIMARY KEY,
    gender                text,
    age                   integer,
    country               text,
    subscription_type     text,
    listening_time        integer,
    songs_played_per_day  integer,
    skip_rate             double precision,
    device_type           text,
    ads_listened_per_week integer,
    offline_listening     integer,
    is_churned            integer
);

COPY raw_users FROM '/data/spotify_churn_dataset.csv' WITH (FORMAT csv, HEADER true);
