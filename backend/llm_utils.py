import json
import os
import time as time_module
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_DEFAULT_MODEL = "gpt-5-mini"
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
OPENAI_REQUEST_TIMEOUT_SECONDS = 45
OPENAI_ATTEMPTS = 3
OPENAI_RETRY_SECONDS = 2.0
ALLOW_LLM_FALLBACK_ENV = "ALLOW_LLM_FALLBACK"


def allow_llm_fallback() -> bool:
    raw = os.getenv(ALLOW_LLM_FALLBACK_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def openai_api_key(required: bool = True) -> str:
    token = os.getenv(OPENAI_API_KEY_ENV, "").strip()
    if token or not required:
        return token
    raise RuntimeError(f"{OPENAI_API_KEY_ENV} is required for LLM analysis")


def openai_model() -> str:
    configured = os.getenv(OPENAI_MODEL_ENV, "").strip()
    return configured or OPENAI_DEFAULT_MODEL


def extract_response_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    fragments = []
    for output_item in payload.get("output", []) or []:
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []) or []:
            if not isinstance(content_item, dict):
                continue
            text_value = content_item.get("text")
            if isinstance(text_value, str) and text_value.strip():
                fragments.append(text_value.strip())

    if fragments:
        return "\n".join(fragments)

    raise RuntimeError("OpenAI response did not include structured text output")


def request_structured_response(
    *,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: Dict[str, Any],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    body = {
        "model": model or openai_model(),
        "input": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }

    request = Request(
        OPENAI_RESPONSES_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {openai_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, OPENAI_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=OPENAI_REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw_text = extract_response_text(payload)
            return json.loads(raw_text)
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenAI HTTP {exc.code}: {body_text[:240]}")
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc

        if attempt < OPENAI_ATTEMPTS:
            time_module.sleep(OPENAI_RETRY_SECONDS * attempt)

    raise RuntimeError(
        f"Unable to complete structured OpenAI response after {OPENAI_ATTEMPTS} attempts: {last_error}"
    )
