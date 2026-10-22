"""Phone setup metadata is local-only and identifies the inference host."""
import os
import sys
from pathlib import Path

import pytest
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lan
import server


def test_phone_setup_reports_host_platform_when_requested_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given an idle phone proxy and a request on the local UI socket.
    monkeypatch.setattr(lan, "phone_urls", lambda: [])
    monkeypatch.setattr(lan, "access_key", lambda: "test-access-key")
    request = Request({"type": "http", "server": ("127.0.0.1", 8008)})
    # When the browser requests phone setup details.
    result = server.lan_status(request)
    # Then startup instructions can follow the host, not the browser's OS.
    expected = "win32" if os.path.exists("/dev/dxg") else sys.platform
    assert result["platform"] == expected
    assert result["local"] is True
    assert result["running"] is False


def test_phone_setup_hides_link_and_key_when_requested_through_lan() -> None:
    # Given a request arriving through the phone-facing socket.
    request = Request({"type": "http", "server": ("127.0.0.1", lan.LAN_PORT)})
    # When the phone asks for desktop setup metadata.
    result = server.lan_status(request)
    # Then neither the access key nor the authenticated QR link is disclosed.
    assert result == {"local": False}
