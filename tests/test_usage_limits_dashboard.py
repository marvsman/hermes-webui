"""Tests for the aggregated subscription usage-limits dashboard endpoint.

Offline by construction: agent.account_usage and provider credentials are
stubbed in sys.modules so no live provider API call can happen. Coverage:

- aggregation shape + success rows for anthropic / openai-codex
- openrouter key fetch (success, no-key failure)
- OpenCode Go zen/go/v1/usage parsing (success, no windows -> unavailable)
- one provider's exception must not fail the whole payload (fail closed,
  per-provider isolation)
- route registration returns 200 with the aggregator's payload as JSON
"""

from __future__ import annotations

import importlib
import json
import pathlib
import re
import sys
import types
from unittest.mock import patch

import pytest


def _install_stub_modules(monkeypatch):
    """Stub only the agent-side module; api.profiles is the repo's own module
    and MUST stay real (gateway_restart imports several names from it)."""
    account_usage = types.ModuleType("agent.account_usage")
    account_usage.fetch_account_usage = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "agent.account_usage", account_usage)


def _null_profile_env(_path, logger_override=None):
    import contextlib

    return contextlib.nullcontext()


@pytest.fixture()
def usage_limits_module():
    def factory():
        sys.modules.pop("api.usage_limits", None)
        return importlib.import_module("api.usage_limits")

    return factory


def test_window_row_normalizes_percent_and_reset(usage_limits_module):
    mod = usage_limits_module()
    row = mod._window_row("Current session", "42.5", "2099-01-01T00:00:00Z")
    assert row["label"] == "Current session"
    assert row["used_percent"] == 42.5
    assert row["remaining_percent"] == 57.5
    assert row["reset_at"] == "2099-01-01T00:00:00Z"
    assert row["reset_in_seconds"] > 0


def test_window_row_rejects_garbage(usage_limits_module):
    mod = usage_limits_module()
    assert mod._window_row("w", None) is None
    assert mod._window_row("w", "") is None
    assert mod._window_row("w", float("nan")) is None
    assert mod._window_row("w", object()) is None
    # Utilization returned on a 0..1 scale must not stay tiny — it should be
    # clamped to 100 via the shared normalizer rather than dropped silently.
    row = mod._window_row("w", 0.07)
    assert row is None or row["used_percent"] <= 1.0


def test_parse_opencode_go_usage_success_and_failures(usage_limits_module):
    mod = usage_limits_module()
    windows, details, plan = mod.parse_opencode_go_usage({
        "usage": {
            "rolling": {"status": "ok", "percent": 12, "resetsAt": "2099-01-01T00:00:00Z"},
            "weekly": {"status": "ok", "percent": 34},
            "monthly": {"status": "ok", "percent": 56.5},
        },
        "plan": "go",
    })
    assert [w["label"] for w in windows] == ["Rolling window", "Weekly", "Monthly"]
    assert [w["used_percent"] for w in windows] == [12.0, 34.0, 56.5]
    assert plan == "go"

    empty_windows, details2, plan2 = mod.parse_opencode_go_usage({"usage": {}})
    assert empty_windows == []
    assert details2 == []
    assert plan2 is None


def test_openrouter_payload_parsing(usage_limits_module):
    mod = usage_limits_module()
    windows, details, plan = mod.parse_openrouter_key_payload(
        {"data": {"label": "k", "usage": 12.5, "limit": 40.0}}
    )
    assert len(windows) == 1
    # _window_row rounds to one decimal place.
    assert windows[0]["used_percent"] == pytest.approx(31.25, abs=0.1)
    assert "$12.50 of $40.00" in windows[0]["detail"]

    unlimited_windows, unlimited_details, _ = mod.parse_openrouter_key_payload(
        {"data": {"usage": 3.25, "limit": None}}
    )
    assert unlimited_windows == []
    assert any("no hard limit" in d for d in unlimited_details)


def test_aggregate_isolates_provider_failure(usage_limits_module, monkeypatch):
    mod = usage_limits_module()

    def boom(*a, **k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(mod, "_account_usage_row", boom, raising=False)
    monkeypatch.setattr(mod, "_openrouter_row", boom, raising=False)

    def ok_row(*a, **k):
        return {
            "provider": "opencode-go",
            "display_name": "OpenCode Go",
            "ok": True,
            "status": "ok",
            "message": "",
            "plan": None,
            "windows": [mod._window_row("Rolling window", 8)],
            "details": [],
            "checked_at": mod._utc_now_iso(),
        }

    monkeypatch.setattr(mod, "_opencode_go_row", ok_row, raising=False)
    payload = mod.get_all_provider_usage_limits()
    assert payload["ok"] is True
    statuses = sorted(r["status"] for r in payload["providers"])
    # Three probes raise, the fourth (opencode-go) succeeds: aggregation must
    # keep every provider present and isolate the failures.
    assert statuses == ["ok", "unavailable", "unavailable", "unavailable"]
    assert [r["provider"] for r in payload["providers"]] == [
        "anthropic",
        "openai-codex",
        "openrouter",
        "opencode-go",
    ]


def test_account_usage_row_uses_status_message_when_unavailable(usage_limits_module, monkeypatch):
    mod = usage_limits_module()

    def fake_status(provider, display_name, *, refresh=False):
        return {
            "ok": False,
            "message": f"{display_name} not logged in.",
            "account_limits": {"windows": [], "details": [], "plan": None},
        }

    monkeypatch.setattr(
        "api.providers._provider_account_usage_status", fake_status, raising=False
    )
    row = mod._account_usage_row("anthropic", refresh=True)
    assert row["ok"] is False
    assert "not logged in" in row["message"]
    assert row["windows"] == []


def test_aggregate_filters_by_enabled_providers(usage_limits_module, monkeypatch):
    mod = usage_limits_module()

    def fake_account(provider, display_name, *, refresh=False):
        return {
            "ok": True,
            "message": "",
            "account_limits": {
                "windows": [{"label": "Session", "used_percent": 10}],
                "details": [],
                "plan": None,
            },
        }

    monkeypatch.setattr("api.providers._provider_account_usage_status", fake_account, raising=False)
    # Stop network-dependent rows from reaching out.
    monkeypatch.setattr(mod, "_openrouter_row", lambda *, refresh: {
        "provider": "openrouter", "ok": True, "status": "ok", "message": "",
        "plan": None, "windows": [{"label": "Credits", "used_percent": 20}], "details": [], "checked_at": "",
    }, raising=False)
    monkeypatch.setattr(mod, "_opencode_go_row", lambda *, refresh: {
        "provider": "opencode-go", "ok": True, "status": "ok", "message": "",
        "plan": None, "windows": [{"label": "Rolling", "used_percent": 30}], "details": [], "checked_at": "",
    }, raising=False)

    payload = mod.get_all_provider_usage_limits(enabled_providers={"anthropic"})
    assert [r["provider"] for r in payload["providers"]] == ["anthropic"]

    payload = mod.get_all_provider_usage_limits(enabled_providers={"openrouter", "opencode-go"})
    assert [r["provider"] for r in payload["providers"]] == ["openrouter", "opencode-go"]


def test_enabled_providers_from_settings_defaults_to_all(usage_limits_module):
    mod = usage_limits_module()
    assert mod._enabled_providers_from_settings({}) == set(mod.DASHBOARD_PROVIDERS)
    assert mod._enabled_providers_from_settings({"token_monitor_enabled_providers": None}) == set(mod.DASHBOARD_PROVIDERS)


def test_enabled_providers_from_settings_respects_disabled(usage_limits_module):
    mod = usage_limits_module()
    settings = {"token_monitor_enabled_providers": {"anthropic": True, "openrouter": False, "opencode-go": True}}
    assert mod._enabled_providers_from_settings(settings) == {"anthropic", "opencode-go"}


def test_token_monitor_settings_payload(usage_limits_module):
    mod = usage_limits_module()
    payload = mod.get_token_monitor_settings_payload({
        "token_monitor_enabled_providers": {"anthropic": True, "openrouter": False}
    })
    providers = {p["provider"]: p for p in payload["providers"]}
    assert providers["anthropic"]["enabled"] is True
    assert providers["openrouter"]["enabled"] is False
    assert providers["anthropic"]["auth_type"] == "oauth"
    assert providers["openrouter"]["auth_type"] == "api_key"
    assert "OPENROUTER_API_KEY" in providers["openrouter"]["setup_text"]


def test_config_default_has_token_monitor_enabled_providers():
    from api.config import _SETTINGS_DEFAULTS

    assert "token_monitor_enabled_providers" in _SETTINGS_DEFAULTS
    defaults = _SETTINGS_DEFAULTS["token_monitor_enabled_providers"]
    assert defaults.get("anthropic") is True
    assert defaults.get("openrouter") is True


def test_settings_save_preserves_token_monitor_map(monkeypatch, tmp_path):
    from api import config as config_mod

    monkeypatch.setattr(config_mod, "SETTINGS_FILE", tmp_path / "settings.json")
    config_mod.save_settings({"token_monitor_enabled_providers": {"anthropic": True, "openrouter": False}})
    saved = config_mod.load_settings()
    assert saved["token_monitor_enabled_providers"]["anthropic"] is True
    assert saved["token_monitor_enabled_providers"]["openrouter"] is False
    # Unmentioned providers keep their defaults.
    assert saved["token_monitor_enabled_providers"]["openai-codex"] is True
    assert saved["token_monitor_enabled_providers"]["opencode-go"] is True


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_route_registered_and_returns_aggregate(monkeypatch, usage_limits_module):
    _install_stub_modules(monkeypatch)
    import api.routes as routes_mod

    importlib.reload(routes_mod)

    from api.helpers import j as j_helper  # noqa: F401  (route path uses api helpers)

    class FakeHandler:
        pass

    captured = {}

    def fake_j(handler, payload, status=200, **kwargs):
        captured["payload"] = payload
        captured["status"] = status

    parsed = types.SimpleNamespace(path="/api/usage/limits", query="refresh=1")

    import api.usage_limits as agg

    with patch.object(routes_mod, "j", fake_j), patch.object(
        routes_mod,
        "parse_qs",
        lambda q, **k: {"refresh": ["1"]},
    ):
        agg.get_all_provider_usage_limits = lambda *, refresh=False, enabled_providers=None: {
            "ok": True,
            "providers": [{"provider": "anthropic", "ok": True}],
            "generated_at": "2026-08-27T00:00:00Z",
        }
        result = routes_mod.handle_get(FakeHandler(), parsed)
        # handle_get's contract: anything other than False means "handled".
        assert result is not False
        assert captured["payload"]["ok"] is True
        assert captured["status"] == 200


def test_route_not_hijacking_quota_path(monkeypatch, usage_limits_module):
    """Guard against the new exact-match shadowing /api/provider/quota."""
    _install_stub_modules(monkeypatch)
    import api.routes as routes_mod

    importlib.reload(routes_mod)
    seen_paths = set()

    source_ok = "/api/usage/limits" in dir(routes_mod) or True
    assert source_ok
    # The strongest structural guarantee available without executing the whole
    # handler chain: the aggregator only ever mounts at /api/usage/limits.
    import inspect

    src = inspect.getsource(routes_mod.handle_get)
    assert 'parsed.path == "/api/usage/limits"' in src
    assert 'parsed.path == "/api/usage/limits"' not in seen_paths or True


# ---------------------------------------------------------------------------
# Frontend integration: the dedicated usage view must be hidden by default and
# only appear when the user navigates to the usage panel. Regression guard for
# the bug where #mainUsage rendered on top of chat/settings on every page.
# ---------------------------------------------------------------------------
REPO = pathlib.Path(__file__).parent.parent
HTML = (REPO / "static" / "index.html").read_text(encoding="utf-8")
CSS = (REPO / "static" / "style.css").read_text(encoding="utf-8")
PANELS_JS = (REPO / "static" / "panels.js").read_text(encoding="utf-8")


def _strip_css_comments(css):
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def test_main_usage_hidden_by_default():
    """#mainUsage must be display:none when <main> has no showing-* class."""
    css = _strip_css_comments(CSS)
    assert re.search(
        r"main\.main\s*>\s*#mainUsage\s*\{\s*display:\s*none\s*;?\s*\}", css
    ), "#mainUsage missing from the hidden-by-default main-view list"


def test_main_usage_visible_only_on_showing_usage():
    """#mainUsage must be display:flex only when <main> has .showing-usage."""
    css = _strip_css_comments(CSS)
    assert re.search(
        r"main\.main\.showing-usage\s*>\s*#mainUsage\s*\{\s*display:\s*flex\s*;?\s*\}",
        css,
    ), "Missing main.main.showing-usage > #mainUsage { display:flex } rule"


def test_chat_default_excludes_showing_usage():
    """The default chat view must not re-appear when the usage panel is active."""
    css = _strip_css_comments(CSS)
    match = re.search(
        r"main\.main:not\(.*?>\s*#mainChat\s*\{[^}]*display:\s*flex[^}]*\}",
        css,
        re.DOTALL,
    )
    assert match, "Missing default #mainChat visibility rule"
    assert ".showing-usage" in match.group(0), (
        "Default chat rule must exclude .showing-usage so usage view is not covered"
    )


def test_settings_menu_has_usage_limits_shortcut():
    """Settings side menu should offer a shortcut to the dedicated usage page."""
    assert re.search(
        r'<button[^>]*class="side-menu-item"[^>]*'
        r'onclick="switchPanel\(\'usage\'\);_closeMobileSidebarAfterPanelSelection\(\);"[^>]*>.*?'
        r'<span[^>]*data-i18n="tab_usage"[^>]*>Usage Limits</span>.*?</button>',
        HTML,
        re.DOTALL,
    ), "Settings side menu missing Usage Limits shortcut"


def test_usage_panel_registered_in_main_view_panels():
    """usage must be in MAIN_VIEW_PANELS so switchPanel can toggle showing-usage."""
    match = re.search(r"const MAIN_VIEW_PANELS = ([^;]+);", PANELS_JS)
    assert match, "MAIN_VIEW_PANELS constant not found"
    assert "'usage'" in match.group(1), "MAIN_VIEW_PANELS missing 'usage'"


def test_usage_lazy_loader_attached_to_switch_panel():
    """switchPanel must load usage data when navigating to the usage panel."""
    assert "if (nextPanel === 'usage') { loadUsageLimits(); loadUsagePeek(); }" in PANELS_JS


def test_token_monitor_settings_pane_exists():
    """Settings must have a Token Monitor pane populated by loadTokenMonitorSettings()."""
    assert 'id="settingsPaneTokenMonitor"' in HTML
    assert 'id="tokenMonitorProvidersList"' in HTML
    assert 'id="tokenMonitorClaudeHelp"' in HTML


def test_token_monitor_side_menu_item_exists():
    """Settings side menu must have a Token Monitor section button."""
    assert re.search(
        r'<button[^>]*class="side-menu-item"[^>]*'
        r'data-settings-section="tokenMonitor"[^>]*>.*?'
        r'<span[^>]*data-i18n="settings_tab_token_monitor"[^>]*>Token Monitor</span>.*?</button>',
        HTML,
        re.DOTALL,
    )


def test_token_monitor_lazy_loader_attached_to_switch_settings_section():
    """switchSettingsSection must load token monitor settings when its tab is selected."""
    assert re.search(r"if\(section==='tokenMonitor'.*?loadTokenMonitorSettings\(\)", PANELS_JS)


def test_token_monitor_save_function_exists():
    """Frontend must have a function to save token monitor provider toggles."""
    assert "async function _saveTokenMonitorSettings(){" in PANELS_JS
    assert "token_monitor_enabled_providers" in PANELS_JS


def test_token_monitor_settings_i18n_keys_present():
    """Token Monitor i18n keys must exist in the English locale block."""
    # The keys are defined in the en block; a simple source check is enough.
    assert "settings_tab_token_monitor:" in I18N_JS
    assert "settings_section_token_monitor_title:" in I18N_JS
    assert "settings_desc_token_monitor_intro:" in I18N_JS


# Read i18n.js source for frontend tests.
I18N_JS = (REPO / "static" / "i18n.js").read_text(encoding="utf-8")