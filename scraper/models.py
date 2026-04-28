from __future__ import annotations

import re
from datetime import date, datetime
from types import NoneType, UnionType
from typing import Literal, Union, get_args, get_origin

from pydantic import BaseModel, Field, field_validator, model_validator


def _is_optional_numeric(annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        args = tuple(arg for arg in get_args(annotation) if arg is not NoneType)
        return len(args) == 1 and args[0] in {int, float}
    return annotation in {int, float}


class ScholarshipRecord(BaseModel):
    name: str
    deadline: date
    amount_usd: int | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Scholarship name must not be empty")
        return normalized

    @field_validator("amount_usd")
    @classmethod
    def validate_amount(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("Scholarship amount must be non-negative")
        return value


class UniversityRecord(BaseModel):
    id: str
    name: str
    url: str
    program_level: Literal["phd", "masters", "undergrad"] = "phd"
    acceptance_rate: float | None = None
    min_gpa: float | None = None
    min_gre_verbal: int | None = None
    min_gre_quant: int | None = None
    requires_sat: bool = False
    min_sat: int | None = None
    ap_classes_req: int | None = None
    lor_count: int | None = None
    scholarships: list[ScholarshipRecord] = Field(default_factory=list)
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    null_fields: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"

    @field_validator("id", "name", "url")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Text fields must not be empty")
        return normalized

    @field_validator("acceptance_rate")
    @classmethod
    def validate_acceptance_rate(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("acceptance_rate must be between 0 and 1")
        return value

    @field_validator("min_gpa")
    @classmethod
    def validate_min_gpa(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 4.0:
            raise ValueError("min_gpa must be between 0 and 4.0")
        return value

    @field_validator("min_gre_verbal", "min_gre_quant")
    @classmethod
    def validate_gre_scores(cls, value: int | None) -> int | None:
        if value is not None and not 130 <= value <= 170:
            raise ValueError("GRE scores must be between 130 and 170")
        return value

    @field_validator("min_sat")
    @classmethod
    def validate_sat_score(cls, value: int | None) -> int | None:
        if value is not None and not 400 <= value <= 1600:
            raise ValueError("SAT score must be between 400 and 1600")
        return value

    @field_validator("ap_classes_req", "lor_count")
    @classmethod
    def validate_counts(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("Count fields must be non-negative")
        return value

    @model_validator(mode="after")
    def populate_null_fields_and_confidence(self) -> UniversityRecord:
        numeric_null_fields: list[str] = []
        for field_name, field_info in type(self).model_fields.items():
            if _is_optional_numeric(field_info.annotation) and getattr(self, field_name) is None:
                numeric_null_fields.append(field_name)

        self.null_fields = numeric_null_fields
        if not numeric_null_fields:
            self.confidence = "high"
        elif len(numeric_null_fields) <= 2:
            self.confidence = "medium"
        else:
            self.confidence = "low"
        return self

    @classmethod
    def from_slug(cls, name: str, url: str) -> UniversityRecord:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return cls(id=slug, name=name, url=url)