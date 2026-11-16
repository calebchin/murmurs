"""Network guard for the local vLLM server's processes (D22, hard rule 4).

murmurs puts this folder first on the server's ``PYTHONPATH``, so Python imports this
module at start-up in the server and in every process it spawns. It makes every socket
connection other than a Unix domain socket, and every name lookup for a remote host,
fail. The server talks to murmurs only over a Unix socket, so nothing it needs is lost;
anything that tries to reach the network (vLLM's start-up connectivity probe, for
example) fails as it would on a disconnected machine.

It covers sockets opened through Python's ``socket`` module, which is how the server's
Python libraries connect. It does not cover native code that opens sockets itself.

This module runs inside the server's own environment, not murmurs': it must use only
the standard library, and it never prints, since the server's output is discarded.
"""

import socket

_LOCAL_HOSTS = frozenset({None, "", "localhost", "127.0.0.1", "::1"})
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


class NetworkBlockedError(OSError):
    """A server process tried to use the network."""


def _connect(self: socket.socket, address: object) -> None:
    if self.family != socket.AF_UNIX:
        raise NetworkBlockedError("network access is blocked for the murmurs LLM server")
    _real_connect(self, address)  # type: ignore[arg-type]


def _connect_ex(self: socket.socket, address: object) -> int:
    if self.family != socket.AF_UNIX:
        raise NetworkBlockedError("network access is blocked for the murmurs LLM server")
    return _real_connect_ex(self, address)  # type: ignore[arg-type]


def _getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
    if host not in _LOCAL_HOSTS:
        raise socket.gaierror("name lookups are blocked for the murmurs LLM server")
    return _real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]


socket.socket.connect = _connect  # type: ignore[method-assign,assignment]
socket.socket.connect_ex = _connect_ex  # type: ignore[method-assign,assignment]
socket.getaddrinfo = _getaddrinfo  # type: ignore[assignment]
