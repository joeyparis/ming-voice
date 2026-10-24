"""Offline address discovery checks for the shared phone proxy."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windows import lan_proxy


def test_lan_ips_keeps_route_when_hostname_is_unresolvable() -> None:
    # Given a usable route and a hostname that DNS cannot resolve.
    with patch.object(lan_proxy.socket, "socket") as socket_factory, \
            patch.object(lan_proxy.socket, "getaddrinfo",
                         side_effect=lan_proxy.socket.gaierror("hostname unavailable")):
        socket_factory.return_value.getsockname.return_value = ("192.168.1.20", 50000)
        # When the proxy discovers addresses.
        ips = lan_proxy.lan_ips()
    # Then the default-route address is still available.
    assert ips == ["192.168.1.20"]


def test_lan_ips_uses_hostname_when_route_is_unavailable() -> None:
    # Given no default route and a resolvable private host address.
    with patch.object(lan_proxy.socket, "socket") as socket_factory, \
            patch.object(lan_proxy.socket, "getaddrinfo", return_value=[
                (lan_proxy.socket.AF_INET, 0, 0, "", ("10.0.0.20", 0)),
            ]):
        socket_factory.return_value.connect.side_effect = OSError("no route")
        # When the proxy discovers addresses.
        ips = lan_proxy.lan_ips()
    # Then hostname discovery still supplies an address.
    assert ips == ["10.0.0.20"]


def test_lan_ips_filters_duplicates_public_and_virtual_addresses() -> None:
    # Given a route and hostname records including unsuitable addresses.
    with patch.object(lan_proxy.socket, "socket") as socket_factory, \
            patch.object(lan_proxy.socket, "getaddrinfo", return_value=[
                (lan_proxy.socket.AF_INET, 0, 0, "", (ip, 0))
                for ip in ["192.168.1.20", "10.0.0.20", "203.0.113.20", "172.20.0.1", "192.168.56.1"]
            ]):
        socket_factory.return_value.getsockname.return_value = ("192.168.1.20", 50000)
        # When the proxy discovers addresses.
        ips = lan_proxy.lan_ips()
    # Then private adapters are deduplicated with the default route first.
    assert ips == ["192.168.1.20", "10.0.0.20"]
