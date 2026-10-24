"""Phone setup metadata is local-only and identifies the inference host."""
import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lan

# Test the actual handler without importing server, which starts real timers.
server_path = Path(__file__).resolve().parents[1] / "server.py"
handler = next(node for node in ast.parse(server_path.read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name == "lan_status")
handler.decorator_list = []  # route registration is not part of these handler unit tests
handler_module = ModuleType("lan_handler_under_test")
handler_module.__dict__.update(lan=lan, os=os, sys=sys, Request=Request)
exec(compile(ast.Module(body=[handler], type_ignores=[]), str(server_path), "exec"), handler_module.__dict__)
lan_status = getattr(handler_module, "lan_status")


@pytest.mark.parametrize("host", [
    ("darwin", False, "darwin"),
    ("linux", False, "linux"),
    ("win32", False, "win32"),
    ("linux", True, "win32"),
])
def test_phone_setup_reports_host_platform_when_requested_locally(
        monkeypatch: pytest.MonkeyPatch, host: tuple[str, bool, str]) -> None:
    # Given an idle phone proxy and a request on the local UI socket.
    platform, wsl, expected = host
    monkeypatch.setattr(handler_module, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(handler_module, "os",
                        SimpleNamespace(path=SimpleNamespace(exists={"/dev/dxg": wsl}.__getitem__)))
    monkeypatch.setattr(lan, "phone_urls", lambda: [])
    monkeypatch.setattr(lan, "access_key", lambda: "test-access-key")
    request = Request({"type": "http", "server": ("127.0.0.1", 8008)})
    # When the browser requests phone setup details.
    result = lan_status(request)
    # Then startup instructions can follow the host, not the browser's OS.
    assert result["platform"] == expected
    assert result["local"] is True
    assert result["running"] is False


def test_phone_setup_hides_link_and_key_when_requested_through_lan() -> None:
    # Given a request arriving through the phone-facing socket.
    request = Request({"type": "http", "server": ("127.0.0.1", lan.LAN_PORT)})
    # When the phone asks for desktop setup metadata.
    result = lan_status(request)
    # Then neither the access key nor the authenticated QR link is disclosed.
    assert result == {"local": False}
