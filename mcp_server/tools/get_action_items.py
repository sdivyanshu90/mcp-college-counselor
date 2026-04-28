from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from errors import ToolError
from mcp_server.db import Database
from mcp_server.tools.evaluate_chances import EvaluateChancesInput, evaluate_chances


class GetActionItemsInput(EvaluateChancesInput):
    pass


class DeadlineItem(BaseModel):
    name: str
    deadline: str
    days_remaining: int


class GetActionItemsOutput(BaseModel):
    gaps: list[str]
    upcoming_deadlines: list[DeadlineItem]
    is_competitive: bool
    university_name: str


async def get_action_items(
    db: Database,
    params: GetActionItemsInput,
) -> GetActionItemsOutput:
    university = await db.get_university(params.target_university_id)
    if university is None:
        raise ToolError(
            code="university_not_found",
            message=f"No data for '{params.target_university_id}'",
        )

    scholarships = await db.get_scholarships(params.target_university_id)
    gap_rows: list[tuple[float, str]] = []

    requirement_specs: tuple[tuple[str, str, float, float | None], ...] = (
        ("GPA", "min_gpa", params.student_gpa, _as_optional_float(university.get("min_gpa"))),
        (
            "GRE verbal",
            "min_gre_verbal",
            float(params.gre_verbal or 0),
            _as_optional_float(university.get("min_gre_verbal")),
        ),
        (
            "GRE quant",
            "min_gre_quant",
            float(params.gre_quant or 0),
            _as_optional_float(university.get("min_gre_quant")),
        ),
        (
            "SAT",
            "min_sat",
            float(params.sat_score or 0),
            _as_optional_float(university.get("min_sat")),
        ),
        (
            "AP classes",
            "ap_classes_req",
            float(params.ap_classes_completed),
            _as_optional_float(university.get("ap_classes_req")),
        ),
        (
            "Letters of recommendation",
            "lor_count",
            float(params.lor_count),
            _as_optional_float(university.get("lor_count")),
        ),
    )

    for field_label, field_name, student_value, requirement_value in requirement_specs:
        if requirement_value is None:
            continue
        program_level = str(university.get("program_level") or "phd")
        if field_name == "min_sat" and program_level != "undergrad" and not bool(university.get("requires_sat", False)):
            continue
        if student_value >= requirement_value:
            continue
        gap = requirement_value - student_value
        gap_rows.append(
            (
                gap,
                (
                    f"{field_label}: your {student_value:g} is {gap:.2f} points/points below "
                    f"{university['name']} minimum of {requirement_value:g}."
                ),
            )
        )

    gap_rows.sort(key=lambda item: item[0], reverse=True)

    today = date.today()
    upcoming_deadlines: list[DeadlineItem] = []
    for scholarship in scholarships:
        deadline_str = scholarship.get("deadline")
        if not isinstance(deadline_str, str):
            continue
        deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()
        days_remaining = (deadline_date - today).days
        if 0 <= days_remaining <= 90:
            name = scholarship.get("name")
            if not isinstance(name, str):
                continue
            upcoming_deadlines.append(
                DeadlineItem(
                    name=name,
                    deadline=deadline_str,
                    days_remaining=days_remaining,
                )
            )

    upcoming_deadlines.sort(key=lambda item: item.deadline)
    chances = await evaluate_chances(db, params)

    return GetActionItemsOutput(
        gaps=[message for _, message in gap_rows],
        upcoming_deadlines=upcoming_deadlines,
        is_competitive=chances.probability > 0.65,
        university_name=str(university["name"]),
    )


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None