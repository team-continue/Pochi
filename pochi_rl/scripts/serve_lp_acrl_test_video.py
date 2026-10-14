"""Serve only matrix video assets on the user's LAN, including byte-range seeks."""

import argparse
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ASSETS = {"index.html", "test_matrix_9x9.mp4", "test_matrix_9x9.jpg", "cases.csv"}


class VideoHandler(SimpleHTTPRequestHandler):
  def send_head(self):
    name = unquote(urlsplit(self.path).path).removeprefix("/") or "index.html"
    if name not in ASSETS:
      self.send_error(404)
      return None
    self.path = "/" + name
    self.remaining = None
    requested = self.headers.get("Range")
    if name != "test_matrix_9x9.mp4" or not requested:
      return super().send_head()
    path = Path(self.directory) / name
    if not path.is_file():
      self.send_error(404)
      return None
    size = path.stat().st_size
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
    if match and (match[1] or match[2]):
      if match[1]:
        start = int(match[1])
        end = min(int(match[2]), size - 1) if match[2] else size - 1
      else:
        start = max(0, size - int(match[2]))
        end = size - 1
    else:
      start, end = -1, -1
    if not 0 <= start <= end < size:
      self.send_response(416)
      self.send_header("Content-Range", f"bytes */{size}")
      self.send_header("Content-Length", "0")
      self.end_headers()
      return None
    file = path.open("rb")
    file.seek(start)
    self.remaining = end - start + 1
    self.send_response(206)
    self.send_header("Content-Type", "video/mp4")
    self.send_header("Accept-Ranges", "bytes")
    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    self.send_header("Content-Length", str(self.remaining))
    self.end_headers()
    return file

  def copyfile(self, source, outputfile):
    try:
      if self.remaining is None:
        return super().copyfile(source, outputfile)
      while self.remaining:
        chunk = source.read(min(65536, self.remaining))
        if not chunk:
          break
        outputfile.write(chunk)
        self.remaining -= len(chunk)
    except (BrokenPipeError, ConnectionResetError):
      pass


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--directory", required=True, type=Path)
  parser.add_argument("--host", default="192.168.201.32")
  parser.add_argument("--port", type=int, default=8085)
  args = parser.parse_args()

  def handler(*a, **kw):
    return VideoHandler(*a, directory=str(args.directory.resolve()), **kw)

  with ThreadingHTTPServer((args.host, args.port), handler) as server:
    print(f"Video ready at http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
  main()
