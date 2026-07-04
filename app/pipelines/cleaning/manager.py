"""
Cleaning pipeline manager: orchestrates all table cleaners.
"""
from __future__ import annotations

import logging
from typing import Any

from .cleaners import (
    CustomersCleaner,
    DealsCleaner,
    LeadsCleaner,
    OrderItemsCleaner,
    OrdersCleaner,
    PaymentsCleaner,
    ProductsCleaner,
    ReviewsCleaner,
    SellersCleaner,
)

logger = logging.getLogger(__name__)


class CleaningManager:
    """Orchestrate cleaning for all tables in sequence."""

    def __init__(self):
        self.cleaners = [
            OrdersCleaner(),
            OrderItemsCleaner(),
            CustomersCleaner(),
            ProductsCleaner(),
            SellersCleaner(),
            ReviewsCleaner(),
            PaymentsCleaner(),
            LeadsCleaner(),
            DealsCleaner(),
        ]

    def run(self) -> dict[str, Any]:
        """Run all cleaners and return summary."""
        logger.info("=" * 60)
        logger.info("Starting Data Cleaning Pipeline")
        logger.info("=" * 60)

        results = []
        total_initial = 0
        total_final = 0

        for cleaner in self.cleaners:
            result = cleaner.run()
            results.append(result)

            if result["status"] == "success":
                total_initial += result["initial_rows"]
                total_final += result["final_rows"]

        # Summary
        success_count = sum(1 for r in results if r["status"] == "success")
        failed_count = len(results) - success_count

        logger.info("=" * 60)
        logger.info("Cleaning Pipeline Complete")
        logger.info(f"  Tables processed: {len(results)}")
        logger.info(f"  Success: {success_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Total rows: {total_initial:,} → {total_final:,}")
        logger.info(f"  Removed: {total_initial - total_final:,}")
        logger.info("=" * 60)

        return {
            "status": "completed" if failed_count == 0 else "partial",
            "tables_processed": len(results),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_initial_rows": total_initial,
            "total_final_rows": total_final,
            "removed_rows": total_initial - total_final,
            "details": results,
        }
