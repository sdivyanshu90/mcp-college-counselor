from __future__ import annotations

from typing import Any

__all__ = ["FinalRecommendation", "run_agent"]


def __getattr__(name: str) -> Any:
	if name == "FinalRecommendation":
		from client.client import FinalRecommendation

		return FinalRecommendation
	if name == "run_agent":
		from client.client import run_agent

		return run_agent
	raise AttributeError(f"module 'client' has no attribute {name!r}")