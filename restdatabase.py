# scripts/reset_database.py
from app.core.database import sync_engine, create_schemas_and_tables
from sqlmodel import SQLModel, text
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_database():
    """Drop all schemas and recreate from SQLModel metadata."""
    
    with sync_engine.begin() as conn:
        logger.warning("🗑️  Dropping schemas...")
        for schema in [
            settings.postgres_schema_raw,
            settings.postgres_schema_processed,
            settings.postgres_schema_predictions,
        ]:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        logger.info("✓ Schemas dropped")
    
    # Recreate with correct dtypes
    logger.info("🔧 Recreating schemas and tables...")
    create_schemas_and_tables(force_recreate=False)
    logger.info("✅ Database reset complete")

if __name__ == "__main__":
    reset_database()
