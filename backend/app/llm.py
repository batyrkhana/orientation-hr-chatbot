import os
import anthropic

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 1024


def call_claude(messages: list[dict], system: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=system,
        messages=messages,
    )
    return response.content[0].text
