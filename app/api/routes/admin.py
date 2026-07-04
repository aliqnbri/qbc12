# # api/routers/admin.py
# from fastapi import APIRouter, BackgroundTasks, HTTPException
# from scripts.load_raw import run_pipeline
# import logging

# router = APIRouter(prefix="/admin", tags=["admin"])
# log = logging.getLogger(__name__)

# _pipeline_running = False

# @router.post("/load-data")
# async def trigger_load(background_tasks: BackgroundTasks):


#     global _pipeline_running
#     if _pipeline_running:
#         raise HTTPException(status_code=409, detail="Pipeline already running")

#     def _run():
#         global _pipeline_running
#         _pipeline_running = True
#         try:
#             run_pipeline()
#             log.info("✓ ETL pipeline completed")
#         except Exception as e:
#             log.error(f"✗ ETL pipeline failed: {e}")
#         finally:
#             _pipeline_running = False

#     background_tasks.add_task(_run)
#     return {"status": "started", "message": "ETL pipeline running in background"}

# @router.get("/load-status")
# async def load_status():
#     return {"running": _pipeline_running}