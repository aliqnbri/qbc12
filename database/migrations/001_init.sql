-- Migration 001: raw and processed schemas for the Olist late-delivery pipeline.
-- Idempotent: safe to re-run.

BEGIN;

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS processed;

-- ---------------------------------------------------------------------------
-- raw schema
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS raw.orders (
    order_id                       VARCHAR(64) PRIMARY KEY,
    customer_id                    VARCHAR(64) NOT NULL,
    order_status                   VARCHAR(32) NOT NULL,
    order_purchase_timestamp       TIMESTAMP   NOT NULL,
    order_approved_at              TIMESTAMP,
    order_delivered_carrier_date   TIMESTAMP,
    order_delivered_customer_date  TIMESTAMP,
    order_estimated_delivery_date  TIMESTAMP   NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_raw_orders_customer_id ON raw.orders (customer_id);

CREATE TABLE IF NOT EXISTS raw.order_items (
    order_id            VARCHAR(64)      NOT NULL,
    order_item_id       INTEGER          NOT NULL,
    product_id          VARCHAR(64)      NOT NULL,
    seller_id           VARCHAR(64)      NOT NULL,
    shipping_limit_date TIMESTAMP        NOT NULL,
    price               DOUBLE PRECISION NOT NULL,
    freight_value       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (order_id, order_item_id)
);
CREATE INDEX IF NOT EXISTS ix_raw_order_items_product_id ON raw.order_items (product_id);
CREATE INDEX IF NOT EXISTS ix_raw_order_items_seller_id  ON raw.order_items (seller_id);

CREATE TABLE IF NOT EXISTS raw.order_payments (
    order_id             VARCHAR(64)      NOT NULL,
    payment_sequential   INTEGER          NOT NULL,
    payment_type         VARCHAR(32),
    payment_installments INTEGER          NOT NULL,
    payment_value        DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE IF NOT EXISTS raw.order_reviews (
    review_id               VARCHAR(64) NOT NULL,
    order_id                VARCHAR(64) NOT NULL,
    review_score            INTEGER     NOT NULL,
    review_comment_title    TEXT,
    review_comment_message  TEXT,
    review_creation_date    TIMESTAMP   NOT NULL,
    review_answer_timestamp TIMESTAMP,
    PRIMARY KEY (review_id, order_id)
);

CREATE TABLE IF NOT EXISTS raw.customers (
    customer_id              VARCHAR(64)  PRIMARY KEY,
    customer_unique_id       VARCHAR(64)  NOT NULL,
    customer_zip_code_prefix VARCHAR(8),
    customer_city            VARCHAR(128) NOT NULL,
    customer_state           VARCHAR(4)   NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_raw_customers_unique_id ON raw.customers (customer_unique_id);

CREATE TABLE IF NOT EXISTS raw.sellers (
    seller_id              VARCHAR(64)  PRIMARY KEY,
    seller_zip_code_prefix VARCHAR(8),
    seller_city            VARCHAR(128) NOT NULL,
    seller_state           VARCHAR(4)   NOT NULL
);

CREATE TABLE IF NOT EXISTS raw.products (
    product_id                 VARCHAR(64) PRIMARY KEY,
    product_category_name      VARCHAR(128),
    product_name_length        INTEGER,
    product_description_length INTEGER,
    product_photos_qty         INTEGER,
    product_weight_g           DOUBLE PRECISION,
    product_length_cm          DOUBLE PRECISION,
    product_height_cm          DOUBLE PRECISION,
    product_width_cm           DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS raw.marketing_qualified_leads (
    mql_id             VARCHAR(64) PRIMARY KEY,
    first_contact_date TIMESTAMP, -- nullable: some source rows have no contact date
    landing_page_id    VARCHAR(64),
    origin             VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS raw.closed_deals (
    mql_id                        VARCHAR(64) PRIMARY KEY,
    seller_id                     VARCHAR(64) NOT NULL,
    sdr_id                        VARCHAR(64),
    sr_id                         VARCHAR(64),
    won_date                      TIMESTAMP   NOT NULL,
    business_segment              VARCHAR(128),
    lead_type                     VARCHAR(64),
    lead_behaviour_profile        VARCHAR(64),
    has_company                   BOOLEAN,
    has_gtin                      BOOLEAN,
    declared_product_catalog_size DOUBLE PRECISION,
    declared_monthly_revenue      DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_raw_closed_deals_seller_id ON raw.closed_deals (seller_id);

CREATE TABLE IF NOT EXISTS raw.product_category_name_translation (
    product_category_name         VARCHAR(128) PRIMARY KEY,
    product_category_name_english VARCHAR(128) NOT NULL
);

-- ---------------------------------------------------------------------------
-- processed schema
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS processed.order_features (
    order_id                 VARCHAR(64) PRIMARY KEY,
    order_purchase_timestamp TIMESTAMP   NOT NULL,
    seller_id                VARCHAR(64),
    customer_unique_id       VARCHAR(64),
    late_delivery            INTEGER,
    price_total              DOUBLE PRECISION,
    freight_total            DOUBLE PRECISION,
    freight_ratio            DOUBLE PRECISION,
    item_count               DOUBLE PRECISION,
    distinct_sellers         DOUBLE PRECISION,
    payment_installments     DOUBLE PRECISION,
    payment_value_total      DOUBLE PRECISION,
    estimated_days           DOUBLE PRECISION,
    shipping_limit_days      DOUBLE PRECISION,
    purchase_hour            DOUBLE PRECISION,
    purchase_weekday         DOUBLE PRECISION,
    purchase_month           DOUBLE PRECISION,
    is_weekend               DOUBLE PRECISION,
    total_weight_g           DOUBLE PRECISION,
    max_weight_g             DOUBLE PRECISION,
    total_volume_cm3         DOUBLE PRECISION,
    avg_photos_qty           DOUBLE PRECISION,
    avg_description_length   DOUBLE PRECISION,
    zip_distance             DOUBLE PRECISION,
    same_state               DOUBLE PRECISION,
    seller_order_count       DOUBLE PRECISION,
    seller_late_rate         DOUBLE PRECISION,
    seller_avg_delay_days    DOUBLE PRECISION,
    customer_order_count     DOUBLE PRECISION,
    customer_state           VARCHAR(4),
    seller_state             VARCHAR(4),
    payment_type             VARCHAR(32),
    product_category         VARCHAR(128),
    created_at               TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_processed_features_ts     ON processed.order_features (order_purchase_timestamp);
CREATE INDEX IF NOT EXISTS ix_processed_features_seller ON processed.order_features (seller_id);

CREATE TABLE IF NOT EXISTS processed.predictions (
    id               SERIAL PRIMARY KEY,
    order_id         VARCHAR(64)      NOT NULL,
    late_probability DOUBLE PRECISION NOT NULL,
    risk_level       VARCHAR(16)      NOT NULL,
    model_name       VARCHAR(128)     NOT NULL,
    model_version    VARCHAR(32)      NOT NULL,
    latency_seconds  DOUBLE PRECISION,
    created_at       TIMESTAMP        NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_processed_predictions_order_id   ON processed.predictions (order_id);
CREATE INDEX IF NOT EXISTS ix_processed_predictions_created_at ON processed.predictions (created_at);

COMMIT;
