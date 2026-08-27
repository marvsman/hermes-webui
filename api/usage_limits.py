"""Aggregated subscription usage limits for the WebUI token-limit dashboard.

One GET returns every configured subscription provider's rate-limit windows so
the browser needs a single request instead of polling /api/provider/quota once
per provider. Reuses Hermes Agent's OAuth account-usage engine (agent.account_usage
via api.providers) for Claude and Codex, OpenRouter's documented key endpoint
for credits, and OpenCode Zen's Go-subscription usage endpoint for OpenCode Go.

Percentages only: Anthropic/OpenAI subscription APIs expose utilization and
reset times, never raw token counts (see PR notes). Fail closed: an
unconfigured provider reports status "no_key" rather than vanishing, except
providers this build does not support at all ("unsupported").
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Providers surfaced on the dashboard, in display order. Cards are skipped
# entirely only when SUPPORTED_PROVIDERS lacks the id — configuration problems
# must stay visible ("fail closed"), so no_key/unavailable still render.
SUPPORTED_PROVIDERS = frozenset({"anthropic", "openai-codex", "openrouter", "opencode-go"})
DASHBOARD_PROVIDERS = ("anthropic", "openai-codex", "openrouter", "opencode-go")

DISPLAY_NAMES = {
    "anthropic": "Claude",
    "openai-codex": "Codex",
    "openrouter": "OpenRouter",
    "opencode-go": "OpenCode Go",
}

PROVIDER_CONNECTION_HELP = {
    "anthropic": {
        "auth_type": "oauth",
        "setup_text": "Claude limits are read from your Hermes Agent OAuth session. Run `hermes auth` (or `hermes model` and pick Claude) in a terminal, then reload this page.",
    },
    "openai-codex": {
        "auth_type": "oauth",
        "setup_text": "Codex limits are read from your Hermes Agent OAuth session. Run `hermes auth` (or `hermes model` and pick Codex) in a terminal, then reload this page.",
    },
    "openrouter": {
        "auth_type": "api_key",
        "env_var": "OPENROUTER_API_KEY",
        "setup_text": "Set the OPENROUTER_API_KEY environment variable (or add it to $HERMES_HOME/.env) and restart the WebUI server.",
    },
    "opencode-go": {
        "auth_type": "api_key",
        "env_var": "OPENCODE_GO_API_KEY",
        "setup_text": "Set the OPENCODE_GO_API_KEY environment variable (or add it to $HERMES_HOME/.env) and restart the WebUI server.",
    },
}

_PROBE_TIMEOUT_SECONDS = 6.0
# Mirror the account-usage cache TTL so both probe paths age equally.
_CACHE_TTL_SECONDS = 45.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _window_row(label: str, used_percent: Any, reset_at: Any = None, detail: Any = None) -> Optional[dict[str, Any]]:
    """Normalize one window into the shared contract; None when unusable."""
    try:
        used = float(str(used_percent).strip())
    except (TypeError, ValueError):
        try:
            used = float(used_percent)
        except (TypeError, ValueError):
            return None
    if used != used:  # NaN
        return None
    used = max(0.0, min(100.0, used))
    reset_iso = None
    reset_seconds = None
    dt = _coerce_datetime(reset_at)
    if dt is not None:
        reset_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        reset_seconds = max(0, int((dt - datetime.now(timezone.utc)).total_seconds()))
    return {
        "label": str(label or "").strip() or "Window",
        "used_percent": round(used, 1),
        "remaining_percent": round(100.0 - used, 1),
        "reset_at": reset_iso,
        "reset_in_seconds": reset_seconds,
        "detail": str(detail).strip() if isinstance(detail, str) and detail.strip() else None,
    }


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _snapshot_to_rows(snapshot: Any) -> tuple[list[dict[str, Any]], list[str], Optional[str], Optional[str]]:
    """Serialize an account-usage-style snapshot into (windows, details, plan, reason)."""
    if snapshot is None:
        return [], [], None, None
    windows: list[dict[str, Any]] = []
    for window in getattr(snapshot, "windows", ()) or ():
        row = _window_row(
            getattr(window, "label", ""),
            getattr(window, "used_percent", None),
            getattr(window, "reset_at", None),
            getattr(window, "detail", None),
        )
        if row is not None:
            windows.append(row)
    details = [
        str(detail).strip()
        for detail in (getattr(snapshot, "details", ()) or ())
        if str(detail).strip()
    ]
    plan = str(getattr(snapshot, "plan", "") or "").strip() or None
    reason = str(getattr(snapshot, "unavailable_reason", "") or "").strip() or None
    return windows, details, plan, reason


def _account_usage_row(provider: str, *, refresh: bool) -> dict[str, Any]:
    """Claude / Codex rows via the shared account-usage engine (cached upstream)."""
    from api.providers import _provider_account_usage_status

    status = _provider_account_usage_status(provider, DISPLAY_NAMES[provider], refresh=refresh)
    limits = status.get("account_limits") or {}
    windows: list[dict[str, Any]] = []
    for window in limits.get("windows") or []:
        row = _window_row(
            window.get("label"),
            window.get("used_percent"),
            window.get("reset_at"),
            window.get("detail"),
        )
        if row is not None:
            windows.append(row)
    ok = bool(status.get("ok")) and bool(windows)
    return {
        "provider": provider,
        "display_name": DISPLAY_NAMES[provider],
        "ok": ok,
        "status": "ok" if ok else "unavailable",
        "message": "" if ok else str(status.get("message") or "").strip(),
        "plan": limits.get("plan"),
        "windows": windows,
        "details": limits.get("details") or [],
        "checked_at": _utc_now_iso(),
    }


def parse_openrouter_key_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], Optional[str]]:
    """Map OpenRouter GET /api/v1/key data into (windows, details, plan)."""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    windows: list[dict[str, Any]] = []
    details: list[str] = []
    usage = data.get("usage")
    limit = data.get("limit")
    label = "Credits" if isinstance(limit, (int, float)) and limit else "Credits (unlimited)"
    if isinstance(usage, (int, float)) and isinstance(limit, (int, float)) and limit > 0:
        row = _window_row(label, (float(usage) / float(limit)) * 100.0)
        if row is not None:
            row["detail"] = f"${float(usage):.2f} of ${float(limit):.2f}"
            windows.append(row)
    elif isinstance(usage, (int, float)):
        details.append(f"Credits spent: ${float(usage):.2f} (no hard limit)")
    for name, value in (
        ("Rate requests", data.get("rate_requests")),
        ("Rate inputs", data.get("rate_inputs")),
    ):
        if isinstance(value, dict) and (value.get("used") is not None or value.get("limit") is not None):
            used, cap = value.get("used"), value.get("limit")
            if isinstance(used, (int, float)) and isinstance(cap, (int, float)) and cap > 0:
                row = _window_row(name.title(), (float(used) / float(cap)) * 100.0)
                if row is not None:
                    windows.append(row)
    return windows, details, None


def _openrouter_row(*, refresh: bool) -> dict[str, Any]:
    from api.providers import _OPENROUTER_KEY_URL, _get_provider_api_key

    def _row(status: str, message: str) -> dict[str, Any]:
        return {
            "provider": "openrouter",
            "display_name": DISPLAY_NAMES["openrouter"],
            "ok": status == "ok",
            "status": status,
            "message": message,
            "plan": None,
            "windows": [],
            "details": [],
            "checked_at": _utc_now_iso(),
        }

    api_key = _get_provider_api_key("openrouter")
    if not api_key:
        return _row("no_key", "Set OPENROUTER_API_KEY on the server to see OpenRouter credits here.")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    try:
        req = urllib.request.Request(_OPENROUTER_KEY_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status = "invalid_key" if exc.code in (401, 403) else "unavailable"
        message = "OpenRouter rejected the configured API key." if status == "invalid_key" else "OpenRouter is temporarily unreachable."
        return _row(status, message)
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, OSError, ValueError):
        return _row("unavailable", "OpenRouter is temporarily unreachable.")
    windows, details, plan = parse_openrouter_key_payload(payload)
    row = _row("ok" if windows or details else "unavailable", "")
    row.update({"windows": windows, "details": details, "plan": plan})
    if not windows and not details:
        row["message"] = "OpenRouter returned no credit information."
    return row


def parse_opencode_go_usage(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], Optional[str]]:
    """Map OpenCode Zen Go-subscription GET /zen/go/v1/usage into (windows, details, plan).

    Observed success shape (verified live 2026-08-27):
    {"usage": {"rolling": {"status": "ok"|"rate-limited", "percent": 12,
     "resetsAt": "2026-08-27T11:38:50.218Z"}, "weekly": {...}, "monthly": {...}}}
    status values other than "ok" (e.g. "rate-limited") do NOT block rendering:
    the percentage carries the truth. Windows are skipped only when the
    percentage is missing/unparseable.
    """
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    labels = (("rolling", "Rolling window"), ("weekly", "Weekly"), ("monthly", "Monthly"))
    windows: list[dict[str, Any]] = []
    details: list[str] = []
    plan = payload.get("plan") if isinstance(payload.get("plan"), str) else None
    for key, label in labels:
        entry = usage.get(key)
        if not isinstance(entry, dict):
            continue
        row = _window_row(entry.get("label") or label, entry.get("percent"), entry.get("resetsAt"))
        if row is None:
            status_text = str(entry.get("status") or "").strip()
            if status_text and status_text != "ok":
                details.append(f"{label}: {status_text}")
            continue
        status_text = str(entry.get("status") or "").strip()
        if status_text and status_text not in ("ok",):
            row["detail"] = f"{label.lower()}: {status_text}"
        windows.append(row)
    return windows, details, plan


def _opencode_go_snapshot(api_key: str, base_url: Optional[str] = None) -> Any:
    """Probe the Zen Go usage endpoint and return an account-usage-style namespace.

    One automatic retry for HTTP 403: some OpenCode edge nodes transiently
    reject valid keys with Forbidden; the immediate retry succeeds (observed
    live 2026-08-27). The API key is never included in any error text.
    """
    base = (base_url or "https://opencode.ai").rstrip("/")
    url = f"{base}/zen/go/v1/usage"
    # The edge rejects requests using Python's default User-Agent with a
    # blanket 403 (verified live 2026-08-27); any explicit UA passes.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "hermes-webui-usage-limits/1.0",
    }

    def _once() -> dict[str, Any]:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        try:
            payload = _once()
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                time.sleep(0.4)
                payload = _once()
            else:
                raise
    except Exception as exc:  # noqa: BLE001 - converted to snapshot reason below
        reason = f"OpenCode Go usage probe failed: {type(exc).__name__}" if not isinstance(exc, urllib.error.HTTPError) else (
            "OpenCode Go rejected the configured API key." if exc.code in (401, 403) else "OpenCode Go is temporarily unreachable."
        )
        return SimpleNamespace(
            provider="opencode-go",
            source="zen_go_usage_api",
            fetched_at=datetime.now(timezone.utc),
            title="OpenCode Go usage",
            plan=None,
            windows=(),
            details=(),
            unavailable_reason=reason,
            available=False,
        )

    windows_raw, details_raw, plan = parse_opencode_go_usage(payload)
    # _window_row already normalized percentages, labels, and reset times into
    # plain dicts; carry them on a private attribute so callers can render
    # without re-parsing the provider payload.
    return SimpleNamespace(
        provider="opencode-go",
        source="zen_go_usage_api",
        fetched_at=datetime.now(timezone.utc),
        title="OpenCode Go usage",
        plan=plan,
        windows=(),
        details=tuple(details_raw),
        unavailable_reason=None if windows_raw else "OpenCode Go returned no usage windows.",
        available=bool(windows_raw),
        _rows=windows_raw,
    )


def _opencode_go_row(*, refresh: bool) -> dict[str, Any]:
    from api.providers import _get_provider_api_key

    def _row(status: str, message: str) -> dict[str, Any]:
        return {
            "provider": "opencode-go",
            "display_name": DISPLAY_NAMES["opencode-go"],
            "ok": status == "ok",
            "status": status,
            "message": message,
            "plan": None,
            "windows": [],
            "details": [],
            "checked_at": _utc_now_iso(),
        }

    api_key = _get_provider_api_key("opencode-go")
    if not api_key:
        return _row("no_key", "Set OPENCODE_GO_API_KEY on the server to see your Go-plan limits here.")

    cache_key = "opencode-go"
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and not refresh and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
            return dict(cached[1])

    snapshot = _opencode_go_snapshot(api_key)
    if getattr(snapshot, "_rows", None):
        row = _row("ok", "")
        row.update({
            "windows": list(snapshot._rows),
            "details": list(snapshot.details),
            "plan": snapshot.plan,
        })
    else:
        row = _row("unavailable", str(getattr(snapshot, "unavailable_reason", "") or "OpenCode Go usage is unavailable."))
    with _cache_lock:
        _cache[cache_key] = (time.monotonic(), dict(row))
    return row


def get_all_provider_usage_limits(*, refresh: bool = False, enabled_providers: set[str] | None = None) -> dict[str, Any]:
    """Aggregate dashboard payload for every supported subscription provider.

    If *enabled_providers* is provided, only providers in that set are probed
    and returned. Disabled providers are omitted entirely from the dashboard so
    users can hide providers they are not monitoring.
    """
    rows: list[dict[str, Any]] = []
    for provider in DASHBOARD_PROVIDERS:
        if enabled_providers is not None and provider not in enabled_providers:
            continue
        try:
            if provider in ("anthropic", "openai-codex"):
                row = _account_usage_row(provider, refresh=refresh)
            elif provider == "openrouter":
                row = _openrouter_row(refresh=refresh)
            else:
                row = _opencode_go_row(refresh=refresh)
        except Exception as exc:  # noqa: BLE001 - one provider must not kill the page
            logger.warning("usage-limits probe failed for %s: %s", provider, exc)
            row = {
                "provider": provider,
                "display_name": DISPLAY_NAMES.get(provider, provider),
                "ok": False,
                "status": "unavailable",
                "message": "Probe error.",
                "plan": None,
                "windows": [],
                "details": [],
                "checked_at": _utc_now_iso(),
            }
        if provider not in SUPPORTED_PROVIDERS:
            continue
        rows.append(row)
    return {
        "ok": any(row["ok"] for row in rows),
        "providers": rows,
        "generated_at": _utc_now_iso(),
    }


def get_token_monitor_settings_payload(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return provider metadata + current enabled state for the Token Monitor settings pane.

    Reads the enabled-provider map from *settings* when supplied, otherwise falls
    back to the default set so the pane always renders even before settings load.
    """
    enabled = _enabled_providers_from_settings(settings)
    providers: list[dict[str, Any]] = []
    for provider in DASHBOARD_PROVIDERS:
        meta = PROVIDER_CONNECTION_HELP.get(provider, {})
        providers.append({
            "provider": provider,
            "display_name": DISPLAY_NAMES.get(provider, provider),
            "enabled": provider in enabled,
            "auth_type": meta.get("auth_type"),
            "env_var": meta.get("env_var"),
            "setup_text": meta.get("setup_text"),
        })
    return {"providers": providers}


def _enabled_providers_from_settings(settings: dict[str, Any] | None = None) -> set[str]:
    """Resolve the set of providers enabled for monitoring from persisted settings."""
    if settings is None:
        try:
            from api.config import load_settings

            settings = load_settings()
        except Exception:
            settings = {}
    mapping = settings.get("token_monitor_enabled_providers") if isinstance(settings, dict) else None
    if not isinstance(mapping, dict):
        return set(DASHBOARD_PROVIDERS)
    return {str(k).strip() for k, v in mapping.items() if v and str(k).strip() in SUPPORTED_PROVIDERS}
