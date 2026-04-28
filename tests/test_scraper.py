from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock
from unittest.mock import AsyncMock

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from tenacity import wait_none

from config import settings
from errors import ScraperBlockedError
from mcp_server.db import Database
from scraper.models import UniversityRecord
from scraper.scraper import UniversityScraper
from scraper.selectors import SelfHealingExtractor


class FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status


class FakePage:
    def __init__(
        self,
        evaluate_handler: Callable[[str, object | None], object],
        goto_outcomes: list[object] | None = None,
        url: str = "https://example.edu/program",
    ) -> None:
        self._evaluate_handler = evaluate_handler
        self._goto_outcomes = list(goto_outcomes or [FakeResponse(200)])
        self.url = url
        self.screenshots: list[str] = []

    async def goto(self, url: str, wait_until: str, timeout: int) -> FakeResponse | None:
        _ = (url, wait_until, timeout)
        outcome = self._goto_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, FakeResponse) else None

    async def evaluate(self, script: str, arg: object | None = None) -> object:
        return self._evaluate_handler(script, arg)

    async def wait_for_timeout(self, milliseconds: int) -> None:
        _ = milliseconds

    async def wait_for_function(self, expression: str, timeout: int | None = None) -> None:
        _ = (expression, timeout)

    async def screenshot(self, path: str, full_page: bool) -> None:
        _ = full_page
        self.screenshots.append(path)


class FakeContext:
    def __init__(self, page: FakePage, user_agents: list[str]) -> None:
        self._page = page
        self._user_agents = user_agents

    async def new_page(self) -> FakePage:
        return self._page

    async def close(self) -> None:
        return None


class FakeBrowser:
    def __init__(self, page_factory: Callable[[], FakePage], user_agents: list[str]) -> None:
        self._page_factory = page_factory
        self._user_agents = user_agents

    async def new_context(self, user_agent: str) -> FakeContext:
        self._user_agents.append(user_agent)
        return FakeContext(self._page_factory(), self._user_agents)

    async def close(self) -> None:
        return None


class FakeChromium:
    def __init__(self, page_factory: Callable[[], FakePage], user_agents: list[str]) -> None:
        self._page_factory = page_factory
        self._user_agents = user_agents

    async def launch(self, headless: bool, args: Sequence[str]) -> FakeBrowser:
        _ = (headless, args)
        return FakeBrowser(self._page_factory, self._user_agents)


class FakePlaywright:
    def __init__(self, page_factory: Callable[[], FakePage], user_agents: list[str]) -> None:
        self.chromium = FakeChromium(page_factory, user_agents)


class FakePlaywrightManager:
    def __init__(self, page_factory: Callable[[], FakePage], user_agents: list[str]) -> None:
        self._page_factory = page_factory
        self._user_agents = user_agents

    async def __aenter__(self) -> FakePlaywright:
        return FakePlaywright(self._page_factory, self._user_agents)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        _ = (exc_type, exc, traceback)
        return None


class FakeOpenAIExtractionClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.calls = 0
        self._payload = payload

    async def create_json_completion(self, **kwargs: Any) -> str:
        _ = kwargs
        self.calls += 1
        return json.dumps(self._payload)


def make_scraper(tmp_path: Path) -> UniversityScraper:
    scraper = UniversityScraper()
    scraper.database = Database(str(tmp_path / "scraper.db"))
    return scraper


def full_extraction_payload() -> dict[str, object]:
    return {
        "acceptance_rate": 0.04,
        "min_gpa": 3.7,
        "min_gre_verbal": 160,
        "min_gre_quant": 167,
        "requires_sat": False,
        "min_sat": None,
        "ap_classes_req": 5,
        "lor_count": 3,
        "scholarships": [],
    }


@pytest.mark.asyncio
async def test_pass_a_extracts_gpa_from_dom_text() -> None:
    extractor = SelfHealingExtractor()

    def evaluate_handler(script: str, arg: object | None) -> object:
        if "innerText" in script and arg is None:
            return "Minimum GPA: 3.7 required\nAcceptance rate: 4%\n3 letters of recommendation"
        if isinstance(arg, list) and "minimum gpa" in arg:
            return ["Minimum GPA: 3.7 required"]
        if isinstance(arg, list) and "acceptance rate" in arg:
            return ["Acceptance rate: 4%"]
        if isinstance(arg, list) and "letters of recommendation" in arg:
            return ["3 letters of recommendation"]
        return []

    page = FakePage(evaluate_handler)
    llm_mock = AsyncMock(return_value={})
    setattr(extractor, "_extract_with_llm", llm_mock)

    result = await extractor.extract(cast(Any, page), "MIT EECS")

    assert result["min_gpa"] == pytest.approx(3.7)
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_pass_b_is_triggered_when_pass_a_extracts_too_few_fields() -> None:
    extractor = SelfHealingExtractor()

    def evaluate_handler(script: str, arg: object | None) -> object:
        if "innerText" in script and arg is None:
            return "Minimum GPA: 3.7 required\nScholarships available"
        if isinstance(arg, list) and "minimum gpa" in arg:
            return ["Minimum GPA: 3.7 required"]
        return []

    page = FakePage(evaluate_handler)
    llm_mock = AsyncMock(
        return_value={
            "acceptance_rate": 0.04,
            "lor_count": 3,
            "requires_sat": False,
            "scholarships": [],
        }
    )
    setattr(extractor, "_extract_with_llm", llm_mock)

    result = await extractor.extract(cast(Any, page), "MIT EECS")

    assert result["acceptance_rate"] == pytest.approx(0.04)
    llm_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_fallback_response_is_cached() -> None:
    extractor = SelfHealingExtractor()

    def evaluate_handler(script: str, arg: object | None) -> object:
        if "innerText" in script and arg is None:
            return "Minimum GPA: 3.7 required\nScholarships available"
        if isinstance(arg, list) and "minimum gpa" in arg:
            return ["Minimum GPA: 3.7 required"]
        return []

    page = FakePage(evaluate_handler)
    llm_mock = AsyncMock(
        return_value={
            "acceptance_rate": 0.04,
            "lor_count": 3,
            "requires_sat": False,
            "scholarships": [],
        }
    )
    setattr(extractor, "_extract_with_llm", llm_mock)

    await extractor.extract(cast(Any, page), "MIT EECS")
    await extractor.extract(cast(Any, page), "MIT EECS")

    llm_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_pass_b_supports_openai_compatible_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "model_provider", "openai")
    extractor = SelfHealingExtractor()

    def evaluate_handler(script: str, arg: object | None) -> object:
        if "innerText" in script and arg is None:
            return "Minimum GPA: 3.7 required\nScholarships available"
        if isinstance(arg, list) and "minimum gpa" in arg:
            return ["Minimum GPA: 3.7 required"]
        return []

    page = FakePage(evaluate_handler)
    openai_client = FakeOpenAIExtractionClient(
        {
            "acceptance_rate": 0.04,
            "min_gpa": 3.7,
            "lor_count": 3,
            "requires_sat": False,
            "scholarships": [],
        }
    )
    setattr(extractor, "_openai_client", openai_client)

    result = await extractor.extract(cast(Any, page), "MIT EECS")

    assert result["acceptance_rate"] == pytest.approx(0.04)
    assert openai_client.calls == 1


@pytest.mark.asyncio
async def test_http_403_raises_blocked_error_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    page = FakePage(lambda script, arg: None, goto_outcomes=[FakeResponse(403)])
    user_agents: list[str] = []
    scraper = make_scraper(tmp_path)
    await scraper.database.connect()
    bound_logger = Mock()
    logger = Mock()
    logger.bind.return_value = bound_logger
    scraper._log = logger
    monkeypatch.setattr(
        "scraper.scraper.async_playwright",
        lambda: FakePlaywrightManager(lambda: page, user_agents),
    )

    with pytest.raises(ScraperBlockedError):
        await scraper.scrape_university(scraper._universities[0])

    await scraper.database.close()
    logger.bind.assert_called_with(university_id="mit-eecs")
    bound_logger.warning.assert_called()
    assert bound_logger.warning.call_args.args[0] == "scrape_blocked"


@pytest.mark.asyncio
async def test_runner_continues_after_blocked_scrape(
    mocker: Any,
    tmp_path: Path,
) -> None:
    scraper = make_scraper(tmp_path)
    scraper._universities = [
        {"id": "mit-eecs", "name": "MIT EECS", "url": "https://example.edu/mit"},
        {"id": "stanford-cs", "name": "Stanford CS", "url": "https://example.edu/stanford"},
    ]
    successful_record = UniversityRecord.from_slug("Stanford CS", "https://example.edu/stanford")
    scrape_mock = mocker.patch.object(
        scraper,
        "scrape_university",
        side_effect=[ScraperBlockedError("blocked"), successful_record],
    )

    await scraper.run()

    assert scrape_mock.await_count == 2


@pytest.mark.asyncio
async def test_retry_decorator_retries_timeouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    timeout_error = PlaywrightTimeoutError("playwright timeout")
    page = FakePage(
        lambda script, arg: None,
        goto_outcomes=[timeout_error, FakeResponse(200)],
    )
    user_agents: list[str] = []
    scraper = make_scraper(tmp_path)
    await scraper.database.connect()
    extract_mock = AsyncMock(return_value=full_extraction_payload())
    setattr(scraper.extractor, "extract", extract_mock)
    monkeypatch.setattr(
        "scraper.scraper.async_playwright",
        lambda: FakePlaywrightManager(lambda: page, user_agents),
    )
    retry_controller = cast(Any, UniversityScraper.scrape_university).retry
    original_wait = retry_controller.wait
    retry_controller.wait = wait_none()

    try:
        result = await scraper.scrape_university(scraper._universities[0])
    finally:
        retry_controller.wait = original_wait
        await scraper.database.close()

    assert result.name == "MIT EECS"


def test_university_record_populates_null_fields() -> None:
    record = UniversityRecord(
        id="sample-university",
        name="Sample University",
        url="https://example.edu",
        min_gpa=None,
        min_gre_verbal=None,
        min_gre_quant=160,
        ap_classes_req=None,
        lor_count=2,
    )

    assert set(record.null_fields) >= {"min_gpa", "min_gre_verbal", "ap_classes_req"}


@pytest.mark.asyncio
async def test_user_agent_is_randomized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    random.seed(0)
    user_agents: list[str] = []
    scraper = make_scraper(tmp_path)
    await scraper.database.connect()
    extract_mock = AsyncMock(return_value=full_extraction_payload())
    setattr(scraper.extractor, "extract", extract_mock)
    monkeypatch.setattr(
        "scraper.scraper.async_playwright",
        lambda: FakePlaywrightManager(lambda: FakePage(lambda script, arg: None), user_agents),
    )

    try:
        for _ in range(10):
            await scraper.scrape_university(scraper._universities[0])
    finally:
        await scraper.database.close()

    assert len(set(user_agents)) > 1