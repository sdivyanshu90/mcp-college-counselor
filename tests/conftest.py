from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import pytest
import pytest_asyncio

from mcp_server.db import Database
from scraper.models import UniversityRecord


SeedUniversityFn = Callable[..., Awaitable[None]]


FIXTURE_UNIVERSITY: dict[str, Any] = {
    "id": "mit-eecs",
    "name": "MIT EECS",
    "url": "https://...",
    "acceptance_rate": 0.04,
    "min_gpa": 3.7,
    "min_gre_verbal": 160,
    "min_gre_quant": 167,
    "requires_sat": False,
    "min_sat": None,
    "ap_classes_req": 5,
    "lor_count": 3,
    "scraped_at": "2024-06-01T00:00:00",
}


@pytest.fixture
def seed_university() -> SeedUniversityFn:
    async def _seed(
        database: Database,
        overrides: dict[str, Any] | None = None,
        scholarships: list[dict[str, Any]] | None = None,
    ) -> None:
        payload = dict(FIXTURE_UNIVERSITY)
        payload.update(overrides or {})
        payload["scholarships"] = scholarships or []
        record = UniversityRecord.model_validate(payload)
        await database.upsert_university(record)

    return _seed


@pytest_asyncio.fixture
async def db(
    tmp_path: Any,
    seed_university: SeedUniversityFn,
) -> AsyncGenerator[Database, None]:
    path = str(tmp_path / "test.db")
    database = Database(path)
    await database.connect()
    await seed_university(database)
    yield database
    await database.close()