from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from errors import ToolError
from mcp_server.db import Database
from mcp_server.tools.evaluate_chances import EvaluateChancesInput, evaluate_chances


SeedUniversityFn = Callable[..., Awaitable[None]]


DEFAULT_STUDENT: dict[str, Any] = {
    "student_gpa": 3.7,
    "sat_score": 1420,
    "gre_verbal": 160,
    "gre_quant": 167,
    "ap_classes_completed": 5,
    "lor_count": 3,
    "target_university_id": "mit-eecs",
}


@pytest.mark.parametrize(
    (
        "student_overrides",
        "university_overrides",
        "expected_probability",
        "expected_confidence",
        "expected_test_component",
    ),
    [
        ({}, {}, 1.0, "high", 1.0),
        ({"student_gpa": 1.5}, {}, 0.7622, "high", 1.0),
        (
            {"gre_verbal": 130, "gre_quant": 130, "ap_classes_completed": 0, "lor_count": 0},
            {},
            0.6386,
            "high",
            0.7955,
        ),
        (
            {"sat_score": None, "gre_verbal": None, "gre_quant": None},
            {"min_gre_verbal": None, "min_gre_quant": None},
            1.0,
            "high",
            1.0,
        ),
        ({"ap_classes_completed": 0}, {}, 0.8, "high", 1.0),
        (
            {"student_gpa": 3.5},
            {"min_gpa": None},
            1.0,
            "medium",
            1.0,
        ),
        (
            {},
            {
                "min_gpa": None,
                "ap_classes_req": None,
                "lor_count": None,
                "min_gre_verbal": None,
                "min_gre_quant": None,
            },
            1.0,
            "low",
            1.0,
        ),
        (
            {"sat_score": None},
            {"requires_sat": True, "min_sat": 1500},
            0.7,
            "high",
            0.0,
        ),
    ],
    ids=[
        "perfect-student",
        "gpa-below-minimum",
        "gre-below-minimum",
        "no-test-required",
        "zero-ap-classes",
        "fallback-gpa-threshold",
        "low-confidence-many-nulls",
        "sat-required-missing-score",
    ],
)
async def test_evaluate_chances_cases(
    db: Database,
    seed_university: SeedUniversityFn,
    student_overrides: dict[str, Any],
    university_overrides: dict[str, Any],
    expected_probability: float,
    expected_confidence: str,
    expected_test_component: float,
) -> None:
    await seed_university(db, university_overrides)
    params = EvaluateChancesInput.model_validate({**DEFAULT_STUDENT, **student_overrides})

    result = await evaluate_chances(db, params)

    assert result.probability == pytest.approx(expected_probability, abs=0.01)
    assert result.confidence == expected_confidence
    assert result.component_scores["test"] == pytest.approx(expected_test_component, abs=0.01)


async def test_evaluate_chances_raises_for_missing_university(db: Database) -> None:
    params = EvaluateChancesInput(
        student_gpa=3.7,
        sat_score=1420,
        gre_verbal=160,
        gre_quant=167,
        ap_classes_completed=5,
        lor_count=3,
        target_university_id="unknown-university",
    )

    with pytest.raises(ToolError) as exc_info:
        await evaluate_chances(db, params)

    assert exc_info.value.code == "university_not_found"