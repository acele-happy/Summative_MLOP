"""Round-robin proxy in front of scaled FloraVision API containers.

Docker's DNS server at 127.0.0.11 returns one A record per replica. Querying it
directly avoids libc returning only a single address.
"""

from __future__ import annotations

import itertools
import os
import random
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request

BACKEND_HOST = os.environ.get("BACKEND_HOST", "api")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8000"))
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8080"))
DNS_SERVER = os.environ.get("DNS_SERVER", "127.0.0.11")

_lock = threading.Lock()
_cycle: itertools.cycle | None = None
_known: tuple[str, ...] = ()


def _encode_name(name: str) -> bytes:
    payload = b""
    for label in name.strip(".").split("."):
        encoded = label.encode("ascii")
        payload += bytes([len(encoded)]) + encoded
    return payload + b"\x00"


def _read_name(buffer: bytes, offset: int) -> tuple[str, int]:
    labels: list[str] = []
    seen: set[int] = set()
    end = offset
    moved = False
    while offset < len(buffer) and offset not in seen:
        seen.add(offset)
        length = buffer[offset]
        if length == 0:
            if not moved:
                end = offset + 1
            break
        if length & 0xC0 == 0xC0:
            if not moved:
                end = offset + 2
                moved = True
            offset = ((length & 0x3F) << 8) | buffer[offset + 1]
            continue
        offset += 1
        labels.append(buffer[offset : offset + length].decode("ascii", errors="ignore"))
        offset += length
        if not moved:
            end = offset
    return ".".join(labels), end


def _dns_a(host: str) -> list[str]:
    query_id = random.randint(0, 65535)
    packet = struct.pack(">HHHHHH", query_id, 0x0100, 1, 0, 0, 0)
    packet += _encode_name(host) + struct.pack(">HH", 1, 1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(1.5)
        sock.sendto(packet, (DNS_SERVER, 53))
        data, _ = sock.recvfrom(2048)
    _query_id, _flags, _qd, answers, _ns, _ar = struct.unpack(">HHHHHH", data[:12])
    offset = 12
    _, offset = _read_name(data, offset)
    offset += 4
    addresses: list[str] = []
    for _ in range(answers):
        _, offset = _read_name(data, offset)
        rtype, _rclass, _ttl, rdlength = struct.unpack(">HHIH", data[offset : offset + 10])
        offset += 10
        rdata = data[offset : offset + rdlength]
        offset += rdlength
        if rtype == 1 and rdlength == 4:
            addresses.append(socket.inet_ntoa(rdata))
    return addresses


def backends() -> str:
    global _cycle, _known
    try:
        addresses = _dns_a(BACKEND_HOST)
    except OSError:
        addresses = []
    if not addresses:
        infos = socket.getaddrinfo(BACKEND_HOST, BACKEND_PORT, socket.AF_INET, socket.SOCK_STREAM)
        addresses = [item[4][0] for item in infos]
    unique = tuple(sorted(set(addresses)))
    if not unique:
        raise RuntimeError(f"No backends resolved for {BACKEND_HOST}")
    with _lock:
        if unique != _known:
            print(f"backends: {', '.join(unique)}", flush=True)
            _known = unique
            _cycle = itertools.cycle(unique)
        assert _cycle is not None
        return next(_cycle)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _proxy(self) -> None:
        try:
            address = backends()
        except Exception as exc:
            payload = f"no backend: {exc}".encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else None
        forwarded = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "transfer-encoding", "connection"}
        }
        request = urllib.request.Request(
            f"http://{address}:{BACKEND_PORT}{self.path}",
            data=body,
            headers=forwarded,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
                status = response.status
                header_items = list(response.headers.items())
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            status = exc.code
            header_items = list(exc.headers.items())
        except Exception as exc:
            payload = f"backend {address} failed: {exc}".encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Backend", address)
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(status)
        for key, value in header_items:
            if key.lower() in {"transfer-encoding", "connection", "content-length"}:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Backend", address)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self._proxy()

    def log_message(self, fmt: str, *args) -> None:
        if self.path.startswith("/static") or self.path == "/status":
            return
        print("%s %s" % (self.address_string(), fmt % args), flush=True)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    print(f"Load balancer listening on {LISTEN_PORT} -> {BACKEND_HOST}:{BACKEND_PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
