from __future__ import annotations

from config import Settings


def test_openrouter_settings_choose_openrouter_credentials() -> None:
    settings = Settings(
        MODEL_PROVIDER="openrouter",
        OPENROUTER_API_KEY="test-openrouter-key",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
    )

    assert settings.openai_compatible_api_key == "test-openrouter-key"
    assert settings.openai_compatible_base_url == "https://openrouter.ai/api/v1"