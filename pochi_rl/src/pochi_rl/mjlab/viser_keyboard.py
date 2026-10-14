"""Held-key command leases and a same-origin Viser/keyboard WebSocket server."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import threading
import time
import uuid
from http import HTTPStatus
from pathlib import Path
from urllib.parse import unquote

KEYS = frozenset("wasdqe")


class HeldKeyController:
  def __init__(self, *, timeout=0.5, clock=time.monotonic):
    self.timeout, self.clock = timeout, clock
    self.lock = threading.RLock()
    self.owner = None
    self.keys = frozenset()
    self.last_seen = 0.0
    self.sequences = {}
    self.blocked = set()
    self.enabled = True
    self.speeds = (0.5, 0.3, 0.6)
    self.telemetry = {"ready": False, "command": (0.0, 0.0, 0.0)}

  def _expire(self):
    if self.owner is not None and self.clock() - self.last_seen >= self.timeout:
      self.owner, self.keys = None, frozenset()

  def receive(self, session, keys, sequence):
    if (
      not isinstance(keys, list)
      or len(keys) > 6
      or any(not isinstance(k, str) or k not in KEYS for k in keys)
      or type(sequence) is not int
      or sequence < 0
    ):
      raise ValueError("Require WASDQE keys and a nonnegative sequence")
    with self.lock:
      self._expire()
      if sequence <= self.sequences.get(session, -1):
        return False
      self.sequences[session] = sequence
      pressed = frozenset(keys)
      if not pressed:
        self.blocked.discard(session)
        if self.owner == session:
          self.owner, self.keys = None, frozenset()
        return True
      if not self.enabled or session in self.blocked:
        return False
      if self.owner not in (None, session):
        return False
      self.owner, self.keys, self.last_seen = session, pressed, self.clock()
      return True

  def disconnect(self, session):
    with self.lock:
      if self.owner == session:
        self.owner, self.keys = None, frozenset()
      self.sequences.pop(session, None)
      self.blocked.discard(session)

  def stop(self):
    with self.lock:
      if self.owner is not None:
        self.blocked.add(self.owner)
      self.owner, self.keys = None, frozenset()

  def command(self):
    with self.lock:
      self._expire()
      if not self.enabled:
        return (0.0, 0.0, 0.0)
      k = self.keys
      return tuple(
        speed * (int(pos in k) - int(neg in k))
        for speed, pos, neg in zip(self.speeds, "waq", "sde", strict=True)
      )

  def snapshot(self, session=None):
    with self.lock:
      self._expire()
      return {
        **self.telemetry,
        "keyboard_enabled": self.enabled,
        "keys": sorted(self.keys),
        "busy": self.owner not in (None, session),
      }


class ViserKeyboardGateway:
  """Serve packaged assets only; relay Viser binary traffic to loopback."""

  def __init__(self, controller, *, host, port, backend_port, javascript):
    self.controller = controller
    self.host, self.port, self.backend_port = host, port, backend_port
    self.javascript = Path(javascript).read_bytes()
    self.ready = threading.Event()
    self.error = None

  def start(self):
    self.thread = threading.Thread(target=self._thread_main, daemon=True)
    self.thread.start()
    if not self.ready.wait(10):
      raise RuntimeError("Keyboard gateway startup timed out")
    if self.error is not None:
      raise self.error

  def _thread_main(self):
    try:
      asyncio.run(self._serve())
    except BaseException as exc:
      self.error = exc
      self.ready.set()

  async def _serve(self):
    import viser
    from websockets import Headers
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve
    from websockets.exceptions import ConnectionClosed
    from websockets.http11 import Response

    assets = Path(viser.__file__).resolve().parent / "client/build"
    self.loop = asyncio.get_running_loop()
    self.closed = asyncio.Event()

    def http_response(connection, request):
      if request.headers.get("Upgrade", "").lower() == "websocket":
        origin = request.headers.get("Origin")
        if origin is not None:
          from urllib.parse import urlparse

          if urlparse(origin).netloc != request.headers.get("Host"):
            return Response(HTTPStatus.FORBIDDEN, "Forbidden", Headers(), b"")
        return None
      path = unquote(request.path.partition("?")[0]).replace("\\", "/")
      if path == "/pochi-keyboard.js":
        data, mime = self.javascript, "application/javascript; charset=utf-8"
      else:
        candidate = (assets / (path.lstrip("/") or "index.html")).resolve()
        if not candidate.is_relative_to(assets) or not candidate.is_file():
          return Response(HTTPStatus.NOT_FOUND, "Not Found", Headers(), b"")
        data = candidate.read_bytes()
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.name == "index.html":
          data = data.replace(
            b"</body>", b'<script src="/pochi-keyboard.js"></script></body>'
          )
      return Response(
        HTTPStatus.OK,
        "OK",
        Headers(
          {
            "Content-Type": mime,
            "Content-Length": str(len(data)),
            "Cache-Control": "no-store",
          }
        ),
        data,
      )

    async def relay(source, target):
      async for message in source:
        await target.send(message)

    async def handler(socket):
      if socket.request.path == "/pochi-control":
        session = uuid.uuid4().hex
        try:
          async for message in socket:
            if not isinstance(message, str) or len(message) > 4096:
              await socket.close(1009, "Invalid control packet")
              break
            try:
              packet = json.loads(message)
              accepted = self.controller.receive(
                session, packet["keys"], packet["sequence"]
              )
            except (ValueError, KeyError, TypeError):
              await socket.close(1008, "Invalid key state")
              break
            await socket.send(
              json.dumps({**self.controller.snapshot(session), "accepted": accepted})
            )
        except ConnectionClosed:
          pass
        finally:
          self.controller.disconnect(session)
        return
      if socket.request.path.partition("?")[0] != "/":
        await socket.close(1008, "Unknown socket")
        return
      try:
        async with connect(
          f"ws://127.0.0.1:{self.backend_port}/",
          subprotocols=[socket.subprotocol] if socket.subprotocol else None,
          max_size=64 * 1024**2,
          compression=None,
        ) as upstream:
          tasks = [
            asyncio.create_task(relay(socket, upstream)),
            asyncio.create_task(relay(upstream, socket)),
          ]
          try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
          finally:
            for task in tasks:
              task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
      except (ConnectionClosed, OSError):
        await socket.close()

    async with serve(
      handler,
      self.host,
      self.port,
      process_request=http_response,
      select_subprotocol=lambda _socket, protocols: protocols[0] if protocols else None,
      max_size=64 * 1024**2,
      compression=None,
    ):
      self.ready.set()
      await self.closed.wait()

  def stop(self):
    self.controller.stop()
    if hasattr(self, "loop"):
      self.loop.call_soon_threadsafe(self.closed.set)
      self.thread.join(timeout=5)
