import pytest

from genai.prompts.loader import PromptNotFoundError, list_prompts, load_prompt


def test_list_prompts_includes_all_personas():
    prompts = list_prompts()
    expected = {
        "hotel_analyst",
        "revenue_analyst",
        "operations_manager",
        "executive_assistant",
        "review_analyst",
        "pricing_expert",
    }
    assert expected.issubset(set(prompts))


def test_load_prompt_returns_version_and_text():
    prompt = load_prompt("hotel_analyst")
    assert prompt.name == "hotel_analyst"
    assert prompt.version == "1.0.0"
    assert "Hotel Analyst" in prompt.text


def test_load_prompt_missing_raises():
    with pytest.raises(PromptNotFoundError):
        load_prompt("does_not_exist")


@pytest.mark.parametrize(
    "name",
    ["revenue_analyst", "operations_manager", "executive_assistant", "review_analyst", "pricing_expert"],
)
def test_all_prompts_load(name):
    prompt = load_prompt(name)
    assert prompt.text.strip()
    assert prompt.version
