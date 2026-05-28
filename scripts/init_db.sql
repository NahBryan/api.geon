-- ============================================================
-- GEoN Platform — PostgreSQL Initialization Script
-- Runs automatically when PostgreSQL container first starts
-- ============================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable trigram search extension (for fuzzy text search)
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Enable statistics extension
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Performance settings for ML workload
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET random_page_cost = '1.1';
ALTER SYSTEM SET checkpoint_completion_target = '0.9';
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = '100';
ALTER SYSTEM SET max_connections = '100';

-- Create read-only reporting role
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'agri_readonly') THEN
    CREATE ROLE agri_readonly;
    GRANT CONNECT ON DATABASE agri_risk_db TO agri_readonly;
    GRANT USAGE ON SCHEMA public TO agri_readonly;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public
      GRANT SELECT ON TABLES TO agri_readonly;
  END IF;
END
$$;

-- Log initialization
DO $$
BEGIN
  RAISE NOTICE 'GEoN database initialized at %', NOW();
END
$$;
