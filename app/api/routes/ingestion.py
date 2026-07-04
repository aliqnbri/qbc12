# app/api/routes/ingestion.py
"""
Ingestion API endpoints:
- Upload large CSV files
- Trigger raw data loading into PostgreSQL
"""

from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

from app.core.config import settings
from app.pipelines.tasks.ingestion_tasks import task_load_raw_data

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

RAW_DATA_DIR = Path(settings.raw_data_dir)  # e.g., data/raw
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# حداکثر حجم فایل (100 MB به صورت پیش‌فرض)
MAX_FILE_SIZE_MB = 100
CHUNK_SIZE = 1024 * 1024  # 1 MB chunks for streaming


# ─────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response after successful file upload."""
    status: str = "success"
    filename: str
    file_size_mb: float
    saved_path: str
    message: str


class LoadDataResponse(BaseModel):
    """Response after loading data into PostgreSQL."""
    status: str
    success_count: int
    failed_count: int
    total_rows_inserted: int
    failed_tables: list[str] = Field(default_factory=list)
    duration_seconds: float | None = None


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def validate_csv_filename(filename: str) -> None:
    """Validate that uploaded file is CSV."""
    if not filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV files are allowed. Got: {filename}"
        )


def save_upload_file_streaming(
    upload_file: UploadFile,
    destination: Path,
    max_size_bytes: int,
) -> int:
    """
    Save uploaded file using streaming to handle large files.
    
    Returns:
        Total bytes written
    
    Raises:
        HTTPException if file exceeds max size
    """
    total_bytes = 0
    
    try:
        with destination.open("wb") as f:
            while chunk := upload_file.file.read(CHUNK_SIZE):
                total_bytes += len(chunk)
                
                # Check size limit
                if total_bytes > max_size_bytes:
                    destination.unlink(missing_ok=True)  # حذف فایل ناقص
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max size: {max_size_bytes / (1024**2):.1f} MB"
                    )
                
                f.write(chunk)
        
        logger.info(
            "File saved: %s (%.2f MB)",
            destination.name,
            total_bytes / (1024**2)
        )
        return total_bytes
    
    except Exception as e:
        # پاک‌سازی در صورت خطا
        destination.unlink(missing_ok=True)
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_csv_file(
    file: UploadFile = File(..., description="CSV file to upload"),
    overwrite: bool = Form(False, description="Overwrite if file exists"),
    max_size_mb: int = Form(MAX_FILE_SIZE_MB, description="Max file size in MB"),
) -> UploadResponse:
    """
    **Upload a CSV file** to the raw data directory.
    
    - Supports **large files** via streaming (chunked upload)
    - Validates file extension
    - Checks file size limits
    - Optionally overwrites existing files
    if not file.filename:
    raise HTTPException(400, "No filename")
    """

    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, "Only CSV files allowed")

    destination = RAW_DATA_DIR / file.filename

    # بررسی وجود فایل
    if destination.exists() and not overwrite:
        raise HTTPException(
            409,
            f"File exists: {file.filename}. Use overwrite=true"
        )


    max_bytes = max_size_mb * 1024 * 1024
    total_bytes = save_file_chunked(file, destination, max_bytes)

    return UploadResponse(
        filename=file.filename,
        file_size_mb=round(total_bytes / (1024**2), 2),
        saved_path=str(destination),
    )
    
    
@router.post("/upload-batch")
async def upload_multiple_csv(
    files: list[UploadFile] = File(...),
    overwrite: bool = Form(False),
    max_size_mb: int = Form(MAX_FILE_SIZE_MB),
    ) -> dict[str, Any]:
            
    results = []
    success = 0
    failed = 0

    for file in files:
        try:
            if not file.filename:
                results.append({"filename": "unknown", "status": "failed", "error": "No filename"})
                failed += 1
                continue
            
            if not file.filename.lower().endswith('.csv'):
                results.append({"filename": file.filename, "status": "failed", "error": "Not CSV"})
                failed += 1
                continue
            
            destination = RAW_DATA_DIR / file.filename
            
            if destination.exists() and not overwrite:
                results.append({"filename": file.filename, "status": "skipped", "error": "File exists"})
                failed += 1
                continue
            
            max_bytes = max_size_mb * 1024 * 1024
            total_bytes = save_file_chunked(file, destination, max_bytes)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "file_size_mb": round(total_bytes / (1024**2), 2),
            })
            success += 1
        
        except Exception as e:
            logger.exception("Batch upload error")
            results.append({
                "filename": file.filename if file.filename else "unknown",
                "status": "failed",
                "error": str(e),
            })
            failed += 1

    return {
        "total": len(files),
        "success_count": success,
        "failed_count": failed,
        "results": results,
    }
    
    
@router.post("/load", response_model=LoadDataResponse)
def load_to_postgres() -> LoadDataResponse:
    start = time.time()

    try:
        logger.info("Starting PostgreSQL ingestion")
        metadata = task_load_raw_data()
        duration = time.time() - start
        
        if metadata.get("failed_count", 0) > 0:
            logger.warning("Some tables failed: %s", metadata.get("failed_tables"))
        
        return LoadDataResponse(
            status=metadata.get("status", "success"),
            success_count=metadata.get("success_count", 0),
            failed_count=metadata.get("failed_count", 0),
            total_rows_inserted=metadata.get("total_rows_inserted", 0),
            failed_tables=metadata.get("failed_tables", []),
            duration_seconds=round(duration, 2),
        )

    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(500, f"Load failed: {str(e)}")


@router.get("/files")
def list_files() -> dict[str, Any]:
    if not RAW_DATA_DIR.exists():
        return {
            "directory": str(RAW_DATA_DIR),
            "files": [],
            "count": 0,
        }

    files = []
    for f in RAW_DATA_DIR.glob("*.csv"):
        stat = f.stat()
        files.append({
            "filename": f.name,
            "size_mb": round(stat.st_size / (1024**2), 2),
            "modified": stat.st_mtime,
        })

    return {
        "directory": str(RAW_DATA_DIR),
        "files": sorted(files, key=lambda x: x["filename"]),
        "count": len(files),
    }


@router.delete("/files/{filename}")
def delete_file(filename: str) -> dict[str, str]:
    if not filename.lower().endswith('.csv'):
        raise HTTPException(400, "Only CSV files")

    path = RAW_DATA_DIR / filename

    if not path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    try:
        path.unlink()
        logger.info("Deleted: %s", filename)
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        logger.exception("Delete failed")
        raise HTTPException(500, f"Delete error: {str(e)}")