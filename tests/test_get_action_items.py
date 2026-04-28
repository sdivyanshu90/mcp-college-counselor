from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import date, timedelta
from typing import Any

from mcp_server.db import Database
from mcp_server.tools.get_action_items import GetActionItemsInput, get_action_items


SeedUniversityFn = Callable[..., Awaitable[None]]


DEFAULT_INPUT: dict[str, Any] = {
    "student_gpa": 3.7,
    "sat_score": 1420,
    "gre_verbal": 160,
    "gre_quant": 167,
    "ap_classes_completed": 5,
    "lor_count": 3,
    "target_university_id": "mit-eecs",
}


async def test_get_action_items_no_gaps_when_student_meets_requirements(
    db: Database,
) -> None:
    result = await get_action_items(db, GetActionItemsInput.model_validate(DEFAULT_INPUT))

    assert result.gaps == []
    assert result.is_competitive is True


async def test_get_action_items_includes_gpa_gap(
    db: Database,
) -> None:
    params = GetActionItemsInput.model_validate({**DEFAULT_INPUT, "student_gpa": 3.4})

    result = await get_action_items(db, params)

    assert result.gaps
    assert "0.30" in result.gaps[0]
    assert "MIT EECS" in result.gaps[0]


async def test_get_action_items_filters_deadlines_within_ninety_days(
    db: Database,
    seed_university: SeedUniversityFn,
) -> None:
    scholarships = [
        {
            "name": "Near-term Fellowship",
            "deadline": (date.today() + timedelta(days=30)).isoformat(),
            "amount_usd": 25000,
        },
        {
            "name": "Far-term Fellowship",
            "deadline": (date.today() + timedelta(days=100)).isoformat(),
            "amount_usd": 30000,
        },
    ]
    await seed_university(db, scholarships=scholarships)

    result = await get_action_items(db, GetActionItemsInput.model_validate(DEFAULT_INPUT))

    assert [item.name for item in result.upcoming_deadlines] == ["Near-term Fellowship"]


async def test_get_action_items_marks_student_competitive_when_probability_is_high(
    db: Database,
) -> None:
    result = await get_action_items(db, GetActionItemsInput.model_validate(DEFAULT_INPUT))

    assert result.is_competitive is True


async def test_get_action_items_marks_student_not_competitive_when_probability_is_low(
    db: Database,
    seed_university: SeedUniversityFn,
) -> None:
    await seed_university(db, {"requires_sat": True, "min_sat": 1500})
    params = GetActionItemsInput.model_validate(
        {
            **DEFAULT_INPUT,
            "student_gpa": 1.5,
            "sat_score": None,
            "ap_classes_completed": 0,
            "lor_count": 0,
        }
    )

    result = await get_action_items(db, params)

    assert result.is_competitive is False