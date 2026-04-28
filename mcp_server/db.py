from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from config import SCHEMA_PATH
from errors import DatabaseError
from scraper.models import UniversityRecord


class Database:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._log = structlog.get_logger(__name__).bind(db_path=db_path)

    async def connect(self) -> None:
        database_dir = Path(self._db_path).parent
        await asyncio.to_thread(database_dir.mkdir, parents=True, exist_ok=True)
        schema_sql = await asyncio.to_thread(SCHEMA_PATH.read_text, encoding="utf-8")
        try:
            self._connection = await aiosqlite.connect(self._db_path)
            self._connection.row_factory = aiosqlite.Row
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.executescript(schema_sql)
            await self._connection.commit()
        except aiosqlite.Error as exc:
            self._log.error("db_connect_failed", error=str(exc))
            raise DatabaseError("Failed to connect to database", exc) from exc

    async def close(self) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            await connection.close()
        except aiosqlite.Error as exc:
            self._log.error("db_close_failed", error=str(exc))
            raise DatabaseError("Failed to close database connection", exc) from exc
        finally:
            self._connection = None

    async def get_university(self, university_id: str) -> dict[str, Any] | None:
        connection = self._require_connection()
        try:
            cursor = await connection.execute(
                (
                    "SELECT id, name, url, program_level, acceptance_rate, min_gpa, min_gre_verbal, "
                    "min_gre_quant, requires_sat, min_sat, ap_classes_req, lor_count, scraped_at "
                    "FROM universities WHERE id = ?"
                ),
                (university_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        except aiosqlite.Error as exc:
            self._log.error("db_get_university_failed", error=str(exc), university_id=university_id)
            raise DatabaseError("Failed to fetch university", exc) from exc

        if row is None:
            return None

        payload = dict(row)
        payload["requires_sat"] = bool(payload.get("requires_sat", 0))
        return payload

    async def get_scholarships(self, university_id: str) -> list[dict[str, Any]]:
        connection = self._require_connection()
        try:
            cursor = await connection.execute(
                (
                    "SELECT id, university_id, name, deadline, amount_usd "
                    "FROM scholarships WHERE university_id = ? ORDER BY deadline ASC"
                ),
                (university_id,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        except aiosqlite.Error as exc:
            self._log.error("db_get_scholarships_failed", error=str(exc), university_id=university_id)
            raise DatabaseError("Failed to fetch scholarships", exc) from exc

        return [dict(row) for row in rows]

    async def upsert_university(self, record: UniversityRecord) -> None:
        connection = self._require_connection()
        try:
            await connection.execute(
                (
                    "INSERT OR REPLACE INTO universities ("
                    "id, name, url, program_level, acceptance_rate, min_gpa, min_gre_verbal, min_gre_quant, "
                    "requires_sat, min_sat, ap_classes_req, lor_count, scraped_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    record.id,
                    record.name,
                    record.url,
                    record.program_level,
                    record.acceptance_rate,
                    record.min_gpa,
                    record.min_gre_verbal,
                    record.min_gre_quant,
                    int(record.requires_sat),
                    record.min_sat,
                    record.ap_classes_req,
                    record.lor_count,
                    record.scraped_at.isoformat(),
                ),
            )
            await connection.execute(
                "DELETE FROM scholarships WHERE university_id = ?",
                (record.id,),
            )
            for scholarship in record.scholarships:
                await connection.execute(
                    (
                        "INSERT INTO scholarships (university_id, name, deadline, amount_usd) "
                        "VALUES (?, ?, ?, ?)"
                    ),
                    (
                        record.id,
                        scholarship.name,
                        scholarship.deadline.isoformat(),
                        scholarship.amount_usd,
                    ),
                )
            await connection.commit()
        except aiosqlite.Error as exc:
            self._log.error("db_upsert_university_failed", error=str(exc), university_id=record.id)
            raise DatabaseError("Failed to upsert university", exc) from exc

    async def insert_scrape_log(
        self,
        university_id: str,
        status: str,
        null_fields: list[str],
    ) -> None:
        connection = self._require_connection()
        try:
            await connection.execute(
                (
                    "INSERT INTO scrape_log (university_id, run_at, status, null_fields) "
                    "VALUES (?, ?, ?, ?)"
                ),
                (
                    university_id,
                    datetime.utcnow().isoformat(),
                    status,
                    json.dumps(null_fields),
                ),
            )
            await connection.commit()
        except aiosqlite.Error as exc:
            self._log.error("db_insert_scrape_log_failed", error=str(exc), university_id=university_id)
            raise DatabaseError("Failed to insert scrape log", exc) from exc

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Database.connect() must be called before use")
        return self._connection