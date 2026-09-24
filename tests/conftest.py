"""No test may reach the internet or a live LLM.

Any outbound connection other than loopback or the DATABASE_URL host fails loudly.
Test doubles live in tests/fakes.py.
"""

import contextlib
import ipaddress
import os
import socket
from typing import Any
from urllib.parse import urlparse

import pytest


def _allowed_hosts() -> set[str]:
    allowed = {"localhost"}
    host = urlparse(os.environ.get("DATABASE_URL", "")).hostname
    if host:
        allowed.add(host)
    ips = set()
    for name in allowed:
        with contextlib.suppress(socket.gaierror):
            ips |= {info[4][0] for info in socket.getaddrinfo(name, None)}
    return allowed | ips


_ALLOWED = _allowed_hosts()
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _check(sock: socket.socket, address: Any) -> None:
    if sock.family == socket.AF_UNIX:
        return
    host = address[0]
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    if host not in _ALLOWED:
        raise RuntimeError(f"test tried to open a network connection to {address!r}")


def _guarded_connect(self: socket.socket, address: Any) -> None:
    _check(self, address)
    return _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    _check(self, address)
    return _real_connect_ex(self, address)


@pytest.fixture(autouse=True, scope="session")
def _no_internet():
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    yield
    socket.socket.connect = _real_connect
    socket.socket.connect_ex = _real_connect_ex
