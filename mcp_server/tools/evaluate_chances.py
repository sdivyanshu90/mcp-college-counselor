from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from errors import ToolError
from mcp_server.db import Database


SCORING_WEIGHTS: dict[str, float] = {
    "gpa": 0.40,
    "test_score": 0.30,
    "ap_classes": 0.20,
    "lor": 0.10,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class EvaluateChancesInput(BaseModel):
    student_gpa: float = Field(ge=0.0, le=4.0)
    sat_score: int | None = Field(default=None, ge=400, le=1600)
    gre_verbal: int | None = Field(default=None, ge=130, le=170)
    gre_quant: int | None = Field(default=None, ge=130, le=170)
    ap_classes_completed: int = Field(ge=0)
    lor_count: int = Field(ge=0)
    target_university_id: str

StudentProfile = EvaluateChancesInput

class EvaluateChancesOutput(BaseModel):
    probability: float
    component_scores: dict[str, float]
    university_name: str
    confidence: Literal["high", "medium", "low"]


def compute_test_score_component(
    student: StudentProfile,
    university: dict[str, Any],
) -> float:
    program_level = str(university.get("program_level") or "phd")
    requires_sat = bool(university.get("requires_sat", False))
    minimum_sat = university.get("min_sat")

    # Undergrad: SAT is the relevant test
    if program_level == "undergrad" or requires_sat:
        if student.sat_score is None:
            return 0.0
        denominator = float(minimum_sat or 1400)
        return clamp(student.sat_score / denominator, 0.0, 1.0)

    # PhD / Masters: GRE is the relevant test
    gre_verbal_minimum = university.get("min_gre_verbal")
    gre_quant_minimum = university.get("min_gre_quant")
    if gre_verbal_minimum is not None or gre_quant_minimum is not None:
        if student.gre_verbal is None or student.gre_quant is None:
            return 0.0
        verbal = clamp(student.gre_verbal / float(gre_verbal_minimum or 155), 0.0, 1.0)
        quant = clamp(student.gre_quant / float(gre_quant_minimum or 158), 0.0, 1.0)
        return (verbal + quant) / 2

    return 1.0


async def evaluate_chances(
    db: Database,
    params: EvaluateChancesInput,
) -> EvaluateChancesOutput:
    university = await db.get_university(params.target_university_id)
    if university is None:
        raise ToolError(
            code="university_not_found",
            message=f"No data for '{params.target_university_id}'",
        )

    gpa_score = clamp(params.student_gpa / float(university.get("min_gpa") or 3.5), 0.0, 1.0)
    test_score = compute_test_score_component(params, university)
    ap_score = clamp(
        params.ap_classes_completed / float(max(int(university.get("ap_classes_req") or 1), 1)),
        0.0,
        1.0,
    )
    lor_score = clamp(
        params.lor_count / float(max(int(university.get("lor_count") or 2), 1)),
        0.0,
        1.0,
    )

    probability = sum(
        score * weight
        for score, weight in zip(
            [gpa_score, test_score, ap_score, lor_score],
            SCORING_WEIGHTS.values(),
            strict=True,
        )
    )

    scored_fields = ["min_gpa", "ap_classes_req", "lor_count"]
    if bool(university.get("requires_sat", False)):
        scored_fields.append("min_sat")
    elif university.get("min_gre_verbal") is not None or university.get("min_gre_quant") is not None:
        scored_fields.extend(["min_gre_verbal", "min_gre_quant"])

    null_count = sum(1 for field_name in scored_fields if university.get(field_name) is None)
    confidence: Literal["high", "medium", "low"]
    if null_count == 0:
        confidence = "high"
    elif null_count <= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return EvaluateChancesOutput(
        probability=round(probability, 4),
        component_scores={
            "gpa": gpa_score,
            "test": test_score,
            "ap_classes": ap_score,
            "lor": lor_score,
        },
        university_name=str(university["name"]),
        confidence=confidence,
    )