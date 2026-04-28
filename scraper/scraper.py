from __future__ import annotations

import asyncio
import json
import random

import httpx
import structlog
from playwright.async_api import (
    Browser,
    BrowserContext,
    Error,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)
from pydantic import ValidationError
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import SCREENSHOT_DIR, SEEDS_PATH, settings
from errors import ScraperBlockedError
from mcp_server.db import Database
from scraper.models import ScholarshipRecord, UniversityRecord
from scraper.selectors import SelfHealingExtractor


USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]


def log_retry_attempt(retry_state: RetryCallState) -> None:
    config = retry_state.args[1] if len(retry_state.args) > 1 else {}
    university_id = config.get("id", "unknown") if isinstance(config, dict) else "unknown"
    error = retry_state.outcome.exception() if retry_state.outcome is not None else None
    sleep_for = retry_state.next_action.sleep if retry_state.next_action is not None else None
    structlog.get_logger(__name__).bind(university_id=university_id).warning(
        "scrape_retry",
        attempt=retry_state.attempt_number,
        sleep_seconds=sleep_for,
        error=str(error) if error is not None else "unknown",
    )


class UniversityScraper:
    def __init__(self) -> None:
        seed_payload = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))
        if not isinstance(seed_payload, list):
            raise ValueError("Seed data must be a JSON list")
        self._universities: list[dict[str, str]] = []
        self._fallback_data: dict[str, dict[str, object]] = {}
        for entry in seed_payload:
            if not isinstance(entry, dict):
                continue
            identifier = entry.get("id")
            name = entry.get("name")
            url = entry.get("url")
            if isinstance(identifier, str) and isinstance(name, str) and isinstance(url, str):
                self._universities.append({"id": identifier, "name": name, "url": url})
                fallback = entry.get("fallback_data")
                if isinstance(fallback, dict):
                    self._fallback_data[identifier] = fallback

        self.database = Database(settings.sqlite_path)
        self.extractor = SelfHealingExtractor()
        self._log = structlog.get_logger(__name__)
        self.used_user_agents: list[str] = []

    async def run(self) -> None:
        await self.database.connect()
        try:
            for index, config in enumerate(self._universities):
                university_id = config["id"]
                log = self._log.bind(university_id=university_id)
                try:
                    await self.scrape_university(config)
                except ScraperBlockedError as exc:
                    log.warning("scrape_blocked", error=exc.message)
                    await self.database.insert_scrape_log(university_id, "failed", [])
                except Exception as exc:
                    log.error("scrape_failed", error=str(exc))
                    await self.database.insert_scrape_log(university_id, "failed", [])

                if index < len(self._universities) - 1:
                    delay = random.uniform(settings.scrape_delay_min, settings.scrape_delay_max)
                    await asyncio.sleep(delay)
        finally:
            await self.database.close()

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((TimeoutError, httpx.HTTPStatusError)),
        before_sleep=log_retry_attempt,
        reraise=True,
    )
    async def scrape_university(self, config: dict[str, str]) -> UniversityRecord:
        university_id = config["id"]
        log = self._log.bind(university_id=university_id)
        browser: Browser | None = None
        context: BrowserContext | None = None
        page: Page | None = None
        selected_user_agent = random.choice(USER_AGENTS)
        self.used_user_agents.append(selected_user_agent)

        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox"],
                )
                context = await browser.new_context(user_agent=selected_user_agent)
                page = await context.new_page()
                response = await page.goto(
                    config["url"],
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                if response is not None and response.status == 403:
                    message = "Received HTTP 403 while scraping"
                    log.warning("scrape_blocked", reason=message)
                    raise ScraperBlockedError(message)
                if "captcha" in page.url.lower() or "blocked" in page.url.lower():
                    message = f"Blocked by anti-bot page at {page.url}"
                    log.warning("scrape_blocked", reason=message)
                    raise ScraperBlockedError(message)

                # Wait for the JS framework to hydrate the DOM before reading content.
                # domcontentloaded fires as soon as the HTML is parsed; React/Next.js
                # pages then inject the real content asynchronously.  We wait until
                # document.body.innerText grows beyond a trivial size, falling back
                # gracefully if the page is unusually slow or sparse.
                try:
                    await page.wait_for_function(
                        "document.body.innerText.length > 500",
                        timeout=15_000,
                    )
                except PlaywrightTimeoutError:
                    log.warning("content_hydration_timeout", university_id=university_id)

                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1500)
                extracted = await self.extractor.extract(page, config["name"])

                # If the live scrape could not populate any field (all null), fall back
                # to the representative values stored in seeds/universities.json.
                # This handles programs that do not publish numeric thresholds online
                # (e.g. elite PhD programs that waive GRE and use holistic review).
                numeric_fields = {
                    "acceptance_rate", "min_gpa", "min_gre_verbal", "min_gre_quant",
                    "min_sat", "ap_classes_req", "lor_count",
                }
                all_null = all(extracted.get(f) is None for f in numeric_fields)
                if all_null:
                    fallback = self._fallback_data.get(university_id, {})
                    if fallback:
                        for key, value in fallback.items():
                            if extracted.get(key) is None:
                                extracted[key] = value
                        log.info("scrape_fallback_applied", fields=list(fallback.keys()))

                try:
                    record = UniversityRecord(
                        id=university_id,
                        name=config["name"],
                        url=config["url"],
                        acceptance_rate=_coerce_optional_float(extracted.get("acceptance_rate")),
                        min_gpa=_coerce_optional_float(extracted.get("min_gpa")),
                        min_gre_verbal=_coerce_optional_int(extracted.get("min_gre_verbal")),
                        min_gre_quant=_coerce_optional_int(extracted.get("min_gre_quant")),
                        requires_sat=bool(extracted.get("requires_sat", False)),
                        min_sat=_coerce_optional_int(extracted.get("min_sat")),
                        ap_classes_req=_coerce_optional_int(extracted.get("ap_classes_req")),
                        lor_count=_coerce_optional_int(extracted.get("lor_count")),
                        scholarships=_coerce_scholarships(extracted.get("scholarships")),
                    )
                except ValidationError:
                    await self._save_parse_error_screenshot(page, university_id)
                    log.error("scrape_parse_error", error="UniversityRecord validation failed")
                    raise

                await self.database.upsert_university(record)
                await self.database.insert_scrape_log(
                    record.id,
                    "success" if record.confidence == "high" else "partial",
                    record.null_fields,
                )
                log.info(
                    "scrape_success",
                    confidence=record.confidence,
                    null_fields=record.null_fields,
                    user_agent=selected_user_agent,
                )
                return record
        except PlaywrightTimeoutError as exc:
            log.warning("scrape_timeout", error=str(exc))
            raise TimeoutError(str(exc)) from exc
        except Error as exc:
            log.error("playwright_error", error=str(exc))
            raise
        finally:
            if context is not None:
                try:
                    await context.close()
                except Error as exc:
                    if "Target page, context or browser has been closed" in str(exc):
                        log.info("context_already_closed", error=str(exc))
                    else:
                        log.error("context_close_failed", error=str(exc))
                        raise
            if browser is not None:
                try:
                    await browser.close()
                except Error as exc:
                    if "Target page, context or browser has been closed" in str(exc):
                        log.info("browser_already_closed", error=str(exc))
                    else:
                        log.error("browser_close_failed", error=str(exc))
                        raise

    async def _save_parse_error_screenshot(self, page: Page, university_id: str) -> None:
        await asyncio.to_thread(SCREENSHOT_DIR.mkdir, parents=True, exist_ok=True)
        screenshot_path = SCREENSHOT_DIR / f"{university_id}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)


def _coerce_optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _coerce_scholarships(value: object) -> list[ScholarshipRecord]:
    if not isinstance(value, list):
        return []
    scholarships: list[ScholarshipRecord] = []
    for entry in value:
        if isinstance(entry, dict):
            scholarships.append(ScholarshipRecord.model_validate(entry))
    return scholarships


async def main() -> None:
    scraper = UniversityScraper()
    await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())