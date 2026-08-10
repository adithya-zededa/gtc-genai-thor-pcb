"""Settings that exist must actually do something.

Three settings in this project were documented, editable in the UI, and
read by nothing: `log_retention`, the SMTP block, and the `thresholds`
block. These pin the wiring so a setting cannot silently go inert again.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
import yaml


def test_every_config_yaml_key_has_a_reader():
    """A key nothing greps for is a knob that silently does nothing."""
    cfg = yaml.safe_load(pathlib.Path("config.yaml").read_text())

    def leaves(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from leaves(value, f"{prefix}.{key}" if prefix else key)
        else:
            yield prefix

    orphans = []
    for key in leaves(cfg):
        leaf = key.split(".")[-1]
        found = subprocess.run(
            ["grep", "-rl", "--include=*.py", leaf,
             "agents", "app", "services", "router", "core", "run.py", "wsgi.py"],
            capture_output=True, text=True,
        )
        if not found.stdout.strip():
            orphans.append(key)

    assert not orphans, f"config.yaml keys with no reader: {orphans}"


class TestThresholds:
    """`thresholds.*` used to be documented as live while the tool hardcoded 10/24/5."""

    def test_config_values_are_used(self, tmp_path, monkeypatch):
        from agents.tools.pcb import analytics

        monkeypatch.setattr(
            analytics,
            "load_camera_config",
            lambda: {},
            raising=False,
        )
        monkeypatch.setattr(
            "services.infrastructure.config.load_camera_config",
            lambda: {
                "thresholds": {
                    "max_defects_per_day": 42,
                    "max_high_severity_per_day": 7,
                    "trend_window_hours": 168,
                }
            },
        )

        resolved = analytics._configured_thresholds()  # pylint: disable=protected-access

        assert resolved == {
            "max_defects_per_day": 42.0,
            "max_high_severity_per_day": 7.0,
            "trend_window_hours": 168.0,
        }

    def test_missing_block_falls_back_to_defaults(self, monkeypatch):
        from agents.tools.pcb import analytics

        monkeypatch.setattr(
            "services.infrastructure.config.load_camera_config", lambda: {}
        )

        resolved = analytics._configured_thresholds()  # pylint: disable=protected-access

        assert resolved["max_defects_per_day"] == 10.0
        assert resolved["trend_window_hours"] == 24.0

    def test_garbage_values_do_not_crash_the_tool(self, monkeypatch):
        from agents.tools.pcb import analytics

        monkeypatch.setattr(
            "services.infrastructure.config.load_camera_config",
            lambda: {"thresholds": {"max_defects_per_day": "not a number"}},
        )

        resolved = analytics._configured_thresholds()  # pylint: disable=protected-access

        assert resolved["max_defects_per_day"] == 10.0

    def test_explicit_argument_still_wins(self, monkeypatch):
        """A caller passing a threshold must override the config."""
        from agents.tools.pcb import analytics

        captured = {}

        def _fake_check(**kwargs):
            captured.update(kwargs)
            return {
                "threshold_exceeded": False,
                "total_defects": 0,
                "high_severity_count": 0,
                "window_hours": kwargs["window_hours"],
                "alerts": [],
            }

        monkeypatch.setattr(
            "services.domains.pcb.defect_store.check_threshold_alerts", _fake_check
        )
        monkeypatch.setattr(
            "services.infrastructure.config.load_camera_config",
            lambda: {"thresholds": {"max_defects_per_day": 99}},
        )

        analytics.tool_check_threshold_alerts(rate_threshold=3, window_hours=6)

        assert captured["rate_threshold"] == 3
        assert captured["window_hours"] == 6


class TestEmailSettings:
    """The Settings page writes SMTP into config.yaml; it must be read back."""

    def test_config_yaml_supplies_smtp(self, monkeypatch):
        from agents.tools import email as email_mod

        for var in ("EMAIL_SMTP_SERVER", "EMAIL_SMTP_PORT", "EMAIL_FROM", "EMAIL_USE_TLS"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "services.infrastructure.config.load_camera_config",
            lambda: {
                "notifications": {
                    "email": {
                        "smtp_server": "mail.factory.local",
                        "smtp_port": 2525,
                        "sender_email": "line3@factory.local",
                        "use_tls": False,
                    }
                }
            },
        )

        settings = email_mod._email_settings()  # pylint: disable=protected-access

        assert settings["host"] == "mail.factory.local"
        assert settings["port"] == 2525
        assert settings["sender"] == "line3@factory.local"
        assert settings["use_tls"] is False

    def test_environment_overrides_config(self, monkeypatch):
        from agents.tools import email as email_mod

        monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.override.example")
        monkeypatch.setenv("EMAIL_SMTP_PORT", "465")
        monkeypatch.setattr(
            "services.infrastructure.config.load_camera_config",
            lambda: {"notifications": {"email": {"smtp_server": "ignored", "smtp_port": 25}}},
        )

        settings = email_mod._email_settings()  # pylint: disable=protected-access

        assert settings["host"] == "smtp.override.example"
        assert settings["port"] == 465

    def test_unreadable_config_still_yields_defaults(self, monkeypatch):
        from agents.tools import email as email_mod

        for var in ("EMAIL_SMTP_SERVER", "EMAIL_SMTP_PORT", "EMAIL_FROM", "EMAIL_USER"):
            monkeypatch.delenv(var, raising=False)

        def _boom():
            raise FileNotFoundError("no config.yaml")

        monkeypatch.setattr(
            "services.infrastructure.config.load_camera_config", _boom
        )

        settings = email_mod._email_settings()  # pylint: disable=protected-access

        assert settings["host"] == "smtp.gmail.com"
        assert settings["port"] == 587
        assert settings["sender"] == "noreply@example.com"
