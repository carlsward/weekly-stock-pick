from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PIPELINE_RUN_ID_ENV = "PIPELINE_RUN_ID"
PIPELINE_CACHE_ROOT_ENV = "PIPELINE_CACHE_ROOT"
PIPELINE_SCOPE_ENV = "PIPELINE_SCOPE"
ENABLE_PERSISTENT_PROVIDER_LEDGER_ENV = "ENABLE_PERSISTENT_PROVIDER_LEDGER"
PIPELINE_PROVIDER_LEDGER_PATH_ENV = "PIPELINE_PROVIDER_LEDGER_PATH"
PIPELINE_BUDGET_DATE_ENV = "PIPELINE_BUDGET_DATE"

RUNTIME_STATE_FILENAME = "runtime_state.json"
DEFAULT_PROVIDER_LEDGER_PATH = Path("provider_budget_ledger.json")
PROVIDER_LEDGER_RETENTION_DAYS = 14


def current_run_id() -> str:
    explicit = os.getenv(PIPELINE_RUN_ID_ENV, "").strip()
    if explicit:
        return explicit
    github_run_id = os.getenv("GITHUB_RUN_ID", "").strip()
    if github_run_id:
        return github_run_id
    return f"local-{date.today().isoformat()}"


def current_scope() -> str:
    scope = os.getenv(PIPELINE_SCOPE_ENV, "").strip()
    return scope or "pipeline"


def set_pipeline_scope(scope: str) -> None:
    os.environ[PIPELINE_SCOPE_ENV] = scope.strip() or "pipeline"


def resolve_pipeline_cache_root() -> Path:
    configured = os.getenv(PIPELINE_CACHE_ROOT_ENV, "").strip()
    path = Path(configured) if configured else Path(".pipeline_cache") / current_run_id()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_runtime_state_path() -> Path:
    return resolve_pipeline_cache_root() / RUNTIME_STATE_FILENAME


def persistent_provider_ledger_enabled() -> bool:
    raw = os.getenv(ENABLE_PERSISTENT_PROVIDER_LEDGER_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def current_budget_date() -> str:
    explicit = os.getenv(PIPELINE_BUDGET_DATE_ENV, "").strip()
    if explicit:
        return explicit
    return datetime.now(timezone.utc).date().isoformat()


def resolve_provider_budget_ledger_path() -> Path:
    configured = os.getenv(PIPELINE_PROVIDER_LEDGER_PATH_ENV, "").strip()
    path = Path(configured) if configured else DEFAULT_PROVIDER_LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _default_runtime_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "run_id": current_run_id(),
        "providers": {},
        "events": [],
    }


def _default_provider_budget_ledger() -> Dict[str, Any]:
    return {
        "version": 1,
        "timezone": "UTC",
        "days": {},
    }


def _load_runtime_state() -> Dict[str, Any]:
    path = resolve_runtime_state_path()
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except FileNotFoundError:
        return _default_runtime_state()
    except (json.JSONDecodeError, OSError):
        return _default_runtime_state()

    if not isinstance(state, dict):
        return _default_runtime_state()
    if state.get("run_id") != current_run_id():
        return _default_runtime_state()
    if not isinstance(state.get("providers"), dict):
        state["providers"] = {}
    if not isinstance(state.get("events"), list):
        state["events"] = []
    return state


def _save_runtime_state(state: Dict[str, Any]) -> None:
    path = resolve_runtime_state_path()
    state["run_id"] = current_run_id()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def _load_provider_budget_ledger() -> Dict[str, Any]:
    path = resolve_provider_budget_ledger_path()
    try:
        with path.open("r", encoding="utf-8") as handle:
            ledger = json.load(handle)
    except FileNotFoundError:
        return _default_provider_budget_ledger()
    except (json.JSONDecodeError, OSError):
        return _default_provider_budget_ledger()

    if not isinstance(ledger, dict):
        return _default_provider_budget_ledger()
    if not isinstance(ledger.get("days"), dict):
        ledger["days"] = {}
    return ledger


def _save_provider_budget_ledger(ledger: Dict[str, Any]) -> None:
    path = resolve_provider_budget_ledger_path()
    ledger["timezone"] = "UTC"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(ledger, handle, ensure_ascii=False, indent=2)


def _provider_state(state: Dict[str, Any], provider: str) -> Dict[str, Any]:
    providers = state.setdefault("providers", {})
    entry = providers.get(provider)
    if isinstance(entry, dict):
        entry.setdefault("used", 0)
        entry.setdefault("limit", None)
        entry.setdefault("categories", {})
        return entry

    entry = {
        "used": 0,
        "limit": None,
        "categories": {},
    }
    providers[provider] = entry
    return entry


def _provider_budget_entry(container: Dict[str, Any], provider: str) -> Dict[str, Any]:
    providers = container.setdefault("providers", {})
    entry = providers.get(provider)
    if isinstance(entry, dict):
        entry.setdefault("used", 0)
        entry.setdefault("limit", None)
        entry.setdefault("categories", {})
        return entry

    entry = {
        "used": 0,
        "limit": None,
        "categories": {},
    }
    providers[provider] = entry
    return entry


def _prune_provider_budget_ledger(ledger: Dict[str, Any]) -> None:
    days = ledger.setdefault("days", {})
    try:
        budget_day = date.fromisoformat(current_budget_date())
    except ValueError:
        budget_day = datetime.now(timezone.utc).date()
    keep_after = budget_day - timedelta(days=PROVIDER_LEDGER_RETENTION_DAYS - 1)

    stale_keys = []
    for key in days:
        try:
            parsed = date.fromisoformat(str(key))
        except ValueError:
            stale_keys.append(key)
            continue
        if parsed < keep_after:
            stale_keys.append(key)

    for key in stale_keys:
        days.pop(key, None)


def _daily_budget_bucket(ledger: Dict[str, Any]) -> Dict[str, Any]:
    _prune_provider_budget_ledger(ledger)
    days = ledger.setdefault("days", {})
    bucket = days.get(current_budget_date())
    if isinstance(bucket, dict):
        bucket.setdefault("providers", {})
        return bucket

    bucket = {"providers": {}}
    days[current_budget_date()] = bucket
    return bucket


def _persist_daily_provider_usage(
    provider: str,
    units: int,
    *,
    category: Optional[str],
    daily_limit: Optional[int],
) -> bool:
    if not persistent_provider_ledger_enabled():
        return True

    ledger = _load_provider_budget_ledger()
    bucket = _daily_budget_bucket(ledger)
    entry = _provider_budget_entry(bucket, provider)
    if daily_limit is not None:
        entry["limit"] = max(0, int(daily_limit))

    configured_limit = entry.get("limit")
    used = int(entry.get("used", 0))
    if isinstance(configured_limit, int) and configured_limit >= 0 and used + units > configured_limit:
        _save_provider_budget_ledger(ledger)
        return False

    entry["used"] = used + units
    if category:
        categories = entry.setdefault("categories", {})
        categories[category] = int(categories.get(category, 0)) + units
    _save_provider_budget_ledger(ledger)
    return True


def unique_strings(values: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def record_runtime_event(
    message: str,
    *,
    provider: Optional[str] = None,
    severity: str = "warn",
    degraded: bool = True,
    scope: Optional[str] = None,
) -> None:
    normalized_message = str(message).strip()
    if not normalized_message:
        return

    effective_scope = scope or current_scope()
    state = _load_runtime_state()
    events = state.setdefault("events", [])
    duplicate = any(
        isinstance(existing, dict)
        and existing.get("scope") == effective_scope
        and existing.get("provider") == provider
        and existing.get("message") == normalized_message
        for existing in events
    )
    if duplicate:
        return

    events.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "scope": effective_scope,
            "provider": provider,
            "severity": severity,
            "degraded": bool(degraded),
            "message": normalized_message,
        }
    )
    _save_runtime_state(state)


def consume_provider_budget(
    provider: str,
    units: int = 1,
    *,
    category: Optional[str] = None,
    daily_limit: Optional[int] = None,
    scope: Optional[str] = None,
) -> bool:
    if units <= 0:
        return True

    state = _load_runtime_state()
    provider_entry = _provider_state(state, provider)
    if daily_limit is not None:
        provider_entry["limit"] = max(0, int(daily_limit))

    configured_limit = provider_entry.get("limit")
    used = int(provider_entry.get("used", 0))
    if isinstance(configured_limit, int) and configured_limit >= 0 and used + units > configured_limit:
        _save_runtime_state(state)
        record_runtime_event(
            f"{provider} budget would exceed {configured_limit} units for this pipeline run.",
            provider=provider,
            degraded=True,
            scope=scope,
        )
        return False

    if not _persist_daily_provider_usage(
        provider,
        units,
        category=category,
        daily_limit=daily_limit,
    ):
        _save_runtime_state(state)
        record_runtime_event(
            f"{provider} budget would exceed {daily_limit} units on {current_budget_date()} across persisted pipeline runs.",
            provider=provider,
            degraded=True,
            scope=scope,
        )
        return False

    provider_entry["used"] = used + units
    if category:
        categories = provider_entry.setdefault("categories", {})
        categories[category] = int(categories.get(category, 0)) + units
    _save_runtime_state(state)
    return True


def _provider_daily_status(provider: str, fallback_limit: Optional[Any]) -> Optional[Dict[str, Any]]:
    if not persistent_provider_ledger_enabled():
        return None

    ledger = _load_provider_budget_ledger()
    bucket = ledger.get("days", {}).get(current_budget_date(), {})
    providers = bucket.get("providers", {}) if isinstance(bucket, dict) else {}
    raw_entry = providers.get(provider) if isinstance(providers, dict) else None
    entry = raw_entry if isinstance(raw_entry, dict) else {}

    limit = entry.get("limit", fallback_limit)
    used = int(entry.get("used", 0))
    if isinstance(limit, int):
        remaining = max(limit - used, 0)
        status = "exhausted" if used >= limit else "healthy"
    else:
        remaining = None
        status = "tracking_only"

    return {
        "date": current_budget_date(),
        "status": status,
        "used": used,
        "limit": limit,
        "remaining": remaining,
        "categories": dict(entry.get("categories", {})),
    }


def _provider_status_for_scope(provider: str, entry: Dict[str, Any], scope: str, events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    limit = entry.get("limit")
    used = int(entry.get("used", 0))
    provider_events = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("provider") == provider
        and event.get("scope") == scope
        and event.get("degraded")
    ]
    if isinstance(limit, int):
        remaining = max(limit - used, 0)
        if used >= limit:
            status = "exhausted"
        elif provider_events:
            status = "degraded"
        else:
            status = "healthy"
    else:
        remaining = None
        status = "degraded" if provider_events else "tracking_only"

    daily_status = _provider_daily_status(provider, limit)
    if isinstance(daily_status, dict) and daily_status.get("status") == "exhausted":
        status = "exhausted"
    elif provider_events and status == "healthy":
        status = "degraded"

    payload = {
        "status": status,
        "used": used,
        "limit": limit,
        "remaining": remaining,
        "categories": dict(entry.get("categories", {})),
    }
    if daily_status is not None:
        payload["daily"] = daily_status
    return payload


def build_data_quality_block(
    *,
    scope: Optional[str] = None,
    extra_reasons: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    effective_scope = scope or current_scope()
    state = _load_runtime_state()
    events = [
        event
        for event in state.get("events", [])
        if isinstance(event, dict) and event.get("scope") == effective_scope
    ]
    event_reasons = [
        str(event.get("message", "")).strip()
        for event in events
        if event.get("degraded")
    ]
    reasons = unique_strings([*(extra_reasons or []), *event_reasons])
    providers = {
        provider: _provider_status_for_scope(provider, entry, effective_scope, events)
        for provider, entry in sorted(state.get("providers", {}).items())
        if isinstance(entry, dict)
    }
    return {
        "status": "degraded" if reasons else "healthy",
        "degraded_reason": "; ".join(reasons[:3]) if reasons else None,
        "reasons": reasons,
        "provider_status": providers,
    }


def attach_data_quality(
    payload: Dict[str, Any],
    *,
    scope: Optional[str] = None,
    extra_reasons: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    enriched = dict(payload)
    enriched["data_quality"] = build_data_quality_block(scope=scope, extra_reasons=extra_reasons)
    return enriched


def extract_payload_degraded_reasons(payload: Any, label: str) -> List[str]:
    if not isinstance(payload, dict):
        return []
    block = payload.get("data_quality")
    if not isinstance(block, dict):
        return []
    if str(block.get("status", "")).strip() != "degraded":
        return []

    labeled: List[str] = []
    reason = block.get("degraded_reason")
    if isinstance(reason, str) and reason.strip():
        labeled.append(f"{label}: {reason.strip()}")

    reasons = block.get("reasons")
    if isinstance(reasons, list):
        labeled.extend(
            f"{label}: {str(item).strip()}"
            for item in reasons
            if str(item).strip()
        )
    return unique_strings(labeled)
