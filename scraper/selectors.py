from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import cast

import structlog
from anthropic import AsyncAnthropic
from playwright.async_api import Page

from config import settings
from llm_provider import OpenAICompatibleClient


CacheValue = tuple[float, dict[str, object]]


class SelfHealingExtractor:
    KEYWORD_MAP: dict[str, list[str]] = {
        "acceptance_rate": [
            "acceptance rate", "admit rate", "admissions rate",
            "applicants admitted", "selectivity", "% admitted", "were admitted",
        ],
        "min_gpa": [
            "minimum gpa", "gpa requirement", "grade point average",
            "cumulative gpa", "undergraduate gpa", "academic record", "gpa of", "gpa must be",
        ],
        "min_gre_verbal": [
            "gre verbal", "verbal reasoning", "gre general",
            "gre scores", "gre requirement", "gre test",
        ],
        "min_gre_quant": [
            "gre quantitative", "quant reasoning", "gre general",
            "gre scores", "quantitative score", "gre test",
        ],
        "min_sat": ["sat score", "minimum sat", "sat required", "standardized test"],
        "ap_classes_req": ["ap class", "advanced placement", "ap courses"],
        "lor_count": [
            "letters of recommendation", "recommendation letter",
            "three letters", "two letters", "referees", "references", "letter of reference",
            "faculty references", "academic references",
        ],
        "scholarship": [
            "scholarship", "fellowship", "funding", "financial aid",
            "assistantship", "stipend", "tuition waiver",
        ],
    }

    def __init__(self) -> None:
        self._client: AsyncAnthropic | None = None
        self._openai_client: OpenAICompatibleClient | None = None
        self._cache: dict[str, CacheValue] = {}
        self._log = structlog.get_logger(__name__)

    async def extract(self, page: Page, university_name: str) -> dict[str, object]:
        university_id = re.sub(r"[^a-z0-9]+", "-", university_name.lower()).strip("-")
        log = self._log.bind(university_id=university_id)

        extracted: dict[str, object] = {
            "requires_sat": False,
            "scholarships": [],
        }
        extracted_count = 0

        for field_name, keywords in self.KEYWORD_MAP.items():
            snippets = await self._find_snippets(page, keywords)
            if not snippets:
                continue

            if field_name == "scholarship":
                scholarships = self._parse_scholarships(snippets)
                if scholarships:
                    extracted["scholarships"] = scholarships
                    extracted_count += 1
                continue

            numeric_value = self._parse_numeric("\n".join(snippets), field_name)
            if numeric_value is None:
                continue
            extracted[field_name] = numeric_value
            if field_name == "min_sat":
                extracted["requires_sat"] = True
            extracted_count += 1

        if extracted_count >= 3:
            return extracted

        cached = self._cache.get(university_id)
        now = time.monotonic()
        if cached is not None and now - cached[0] < settings.extraction_cache_ttl:
            return self._merge_results(extracted, cached[1])

        visible_text_value = await page.evaluate(
            "(document.querySelector('main') || "
            "document.querySelector('article') || "
            "document.body).innerText"
        )
        visible_text = cast(str, visible_text_value)
        truncated_text = self._truncate_visible_text(visible_text)

        if not truncated_text:
            return extracted

        try:
            llm_result = await self._extract_with_llm(university_name, truncated_text)
        except Exception as exc:
            log.warning("llm_extraction_failed", error=str(exc))
            return extracted

        merged = self._merge_results(extracted, llm_result)
        self._cache[university_id] = (now, merged)
        return merged

    async def _find_snippets(self, page: Page, keywords: list[str]) -> list[str]:
        result = await page.evaluate(
            """
            (keywords) => {
              const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
              const blockTags = new Set([
                    "P", "DIV", "SECTION", "LI", "TD", "ARTICLE",
                    "MAIN", "ASIDE", "H1", "H2", "H3", "H4", "DT", "DD", "BLOCKQUOTE"
                ]);
              const matches = [];

              while (walker.nextNode()) {
                const node = walker.currentNode;
                const text = (node.textContent || "").trim();
                if (!text) {
                  continue;
                }

                const lowered = text.toLowerCase();
                if (!keywords.some((keyword) => lowered.includes(keyword))) {
                  continue;
                }

                let element = node.parentElement;
                while (element && !blockTags.has(element.tagName)) {
                  element = element.parentElement;
                }

                if (!element) {
                  continue;
                }

                const snippet = (element.innerText || "").trim().slice(0, 400);
                if (snippet) {
                  matches.push(snippet);
                }
              }

              return Array.from(new Set(matches)).slice(0, 8);
            }
            """,
            keywords,
        )
        return [item for item in cast(Sequence[object], result) if isinstance(item, str)]

    async def _extract_with_llm(
        self,
        university_name: str,
        truncated_text: str,
    ) -> dict[str, object]:
        system_prompt = "You are a structured data extractor. Return ONLY valid JSON, no prose."
        user_prompt = (
            f"Extract admission requirements for {university_name} from this text.\n"
            "Return a JSON object with these keys and types:\n"
            "  acceptance_rate: float | null  (0.0 to 1.0, convert percentages),\n"
            "  min_gpa: float | null,\n"
            "  min_gre_verbal: int | null,\n"
            "  min_gre_quant: int | null,\n"
            "  requires_sat: bool,\n"
            "  min_sat: int | null,\n"
            "  ap_classes_req: int | null,\n"
            "  lor_count: int | null,\n"
            "  scholarships: [{name: str, deadline: 'YYYY-MM-DD', amount_usd: int|null}]\n"
            "Use null for any field not mentioned. Never invent data.\n\n"
            f"Text:\n{truncated_text}"
        )

        if settings.model_provider in {"openai", "openrouter"}:
            if self._openai_client is None:
                self._openai_client = OpenAICompatibleClient(
                    api_key=settings.openai_compatible_api_key,
                    base_url=settings.openai_compatible_base_url,
                )
            json_payload = await self._openai_client.create_json_completion(
                model=settings.llm_extraction_model,
                system=system_prompt,
                user_prompt=user_prompt,
                max_completion_tokens=800,
            )
        else:
            if self._client is None:
                self._client = AsyncAnthropic(api_key=settings.anthropic_api_key or None)

            response = await self._client.messages.create(
                model=settings.llm_extraction_model,
                max_tokens=800,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )
            json_payload = "\n".join(
                block.text
                for block in response.content
                if getattr(block, "type", "") == "text" and hasattr(block, "text")
            ).strip()

        parsed = json.loads(json_payload)
        if not isinstance(parsed, dict):
            raise ValueError("LLM extractor response was not a JSON object")
        return self._normalize_llm_result(parsed)

    def _normalize_llm_result(self, payload: dict[str, object]) -> dict[str, object]:
        normalized: dict[str, object] = {"requires_sat": bool(payload.get("requires_sat", False))}
        for field_name in (
            "acceptance_rate",
            "min_gpa",
            "min_gre_verbal",
            "min_gre_quant",
            "min_sat",
            "ap_classes_req",
            "lor_count",
        ):
            value = payload.get(field_name)
            if value is None:
                normalized[field_name] = None
                continue
            if not isinstance(value, (int, float, str)):
                normalized[field_name] = None
                continue
            if field_name in {"acceptance_rate", "min_gpa"}:
                numeric_value = float(value)
                normalized[field_name] = (
                    numeric_value / 100
                    if field_name == "acceptance_rate" and numeric_value > 1
                    else numeric_value
                )
            else:
                normalized[field_name] = int(value)

        scholarships_value = payload.get("scholarships", [])
        normalized_scholarships: list[dict[str, object]] = []
        if isinstance(scholarships_value, list):
            for entry in scholarships_value:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                deadline = entry.get("deadline")
                amount_usd = entry.get("amount_usd")
                if not isinstance(name, str) or not isinstance(deadline, str):
                    continue
                scholarship: dict[str, object] = {
                    "name": name.strip(),
                    "deadline": deadline.strip(),
                    "amount_usd": int(amount_usd) if isinstance(amount_usd, (int, float)) else None,
                }
                normalized_scholarships.append(scholarship)
        normalized["scholarships"] = normalized_scholarships
        return normalized

    def _merge_results(
        self,
        pass_a_result: dict[str, object],
        llm_result: dict[str, object],
    ) -> dict[str, object]:
        merged = dict(llm_result)
        merged.update(pass_a_result)
        if pass_a_result.get("scholarships"):
            merged["scholarships"] = pass_a_result["scholarships"]
        elif "scholarships" not in merged:
            merged["scholarships"] = []
        return merged

    def _truncate_visible_text(self, text: str) -> str:
        stripped = text.strip()
        if len(stripped) <= 6000:
            return stripped
        truncated = stripped[:6000]
        paragraph_break = truncated.rfind("\n\n")
        if paragraph_break > 0:
            return truncated[:paragraph_break].strip()
        return truncated.strip()

    def _parse_scholarships(self, snippets: Sequence[str]) -> list[dict[str, object]]:
        scholarships: list[dict[str, object]] = []
        deadline_pattern = re.compile(r"(20\d{2}-\d{2}-\d{2})")
        amount_pattern = re.compile(r"\$\s*([\d,]+)")

        for snippet in snippets:
            deadline_match = deadline_pattern.search(snippet)
            if deadline_match is None:
                continue
            amount_match = amount_pattern.search(snippet)
            scholarships.append(
                {
                    "name": snippet.splitlines()[0].strip()[:120],
                    "deadline": deadline_match.group(1),
                    "amount_usd": (
                        int(amount_match.group(1).replace(",", ""))
                        if amount_match is not None
                        else None
                    ),
                }
            )
        return scholarships

    def _parse_numeric(self, text: str, field: str) -> float | int | None:
        # lor_count needs both numeric and word-form handling because pages commonly
        # say "three letters" rather than "3 letters".
        if field == "lor_count":
            num_match = re.search(r"\b([2-5])\b", text, flags=re.IGNORECASE)
            if num_match:
                return int(num_match.group(1))
            for word, value in (("five", 5), ("four", 4), ("three", 3), ("two", 2)):
                if re.search(r"\b" + word + r"\b", text, flags=re.IGNORECASE):
                    return value
            return None

        patterns: dict[str, str] = {
            "acceptance_rate": r"(\d+\.?\d*)\s*%",
            "min_gpa": r"(\d\.\d+)",
            # Range-constrained patterns: the snippet is already keyword-filtered,
            # so we further restrict to valid score ranges to avoid grabbing
            # incidental numbers (room numbers, years, phone extensions, etc.).
            "min_gre_verbal": r"verbal[^\d]{0,40}\b(1[3-6]\d|170)\b",
            "min_gre_quant": r"(?:quant|quantitative)[^\d]{0,40}\b(1[3-6]\d|170)\b",
            "min_sat": r"\b([4-9]\d{2}|1[0-5]\d{2}|1600)\b",
            # ap_classes_req: match 1-99, word-boundary prevents partial matches
            # inside longer numbers like years.
            "ap_classes_req": r"\b([1-9]\d?)\b",
        }
        pattern = patterns.get(field)
        if pattern is None:
            return None
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            return None
        if field in {"acceptance_rate", "min_gpa"}:
            fvalue = float(match.group(1))
            return fvalue / 100 if field == "acceptance_rate" else fvalue
        return int(match.group(1))