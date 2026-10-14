"""Render every V18 case into a synchronized 9x9 movie from captured qpos."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="3", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")

import mujoco
import numpy as np
from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
from mujoco.egl import egl_ext as EGL
from OpenGL import GL
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 3600, 2160
CELL_W, CELL_H, LABEL_H = 400, 240, 56
FPS = 25


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def egl_gpu3():
  # EGL enumeration includes software/DRM devices; CUDA_VISIBLE_DEVICES alone
  # does not make EGL index zero correct. Select the visible CUDA device zero.
  query = ctypes.CFUNCTYPE(
    ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_long)
  )(EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
  candidates = []
  for index, device in enumerate(EGL.eglQueryDevicesEXT()):
    value = ctypes.c_long(-1)
    if query(device, 0x323A, ctypes.byref(value)) and value.value == 0:
      candidates.append(index)
  if len(candidates) != 1:
    raise RuntimeError(
      f"Cannot uniquely identify physical GPU3 EGL device: {candidates}"
    )
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(candidates[0])
  return candidates[0]


def command_at(segments, elapsed):
  for start, end, command in segments:
    if start <= elapsed < end:
      return command
  return segments[-1][2]


def vector_line(prefix, values):
  return f"{prefix} x{values[0]:+.2f} y{values[1]:+.2f} w{values[2]:+.2f}"


def load_suite(root, name):
  report = json.loads((root / f"{name}.json").read_text())
  for key in ("states", "model", "visual_fields"):
    assert digest(report[key]) == report[f"{key}_sha256"]
  with np.load(report["states"]) as archive:
    states = {key: archive[key] for key in archive.files}
  with np.load(report["visual_fields"]) as archive:
    fields = {key: archive[key] for key in archive.files}
  model = mujoco.MjModel.from_binary_path(report["model"])
  return {
    "name": name,
    "report": report,
    "states": states,
    "fields": fields,
    "model": model,
  }


def cells_for(suites, layout):
  cells = []
  for suite in suites:
    report = suite["report"]
    indices = (
      range(len(report["profiles"]))
      if layout == "cases"
      else range(len(report["names"]))
    )
    for index in indices:
      profile = report["names"][index]
      cells.append(
        {
          "suite": suite,
          "index": index,
          "profile": profile,
          "trial": index // len(report["profiles"]) + 1,
          "segments": report["profiles"][profile],
          "passed": report["episodes"][index]["passed"],
        }
      )
  return cells


def make_labels(page, page_number, pages, plan, font, small, layout):
  layer = Image.new("RGB", (WIDTH, HEIGHT), (12, 15, 20))
  draw = ImageDraw.Draw(layer)
  for slot, cell in enumerate(page):
    x, y = slot % 9 * CELL_W, slot // 9 * CELL_H
    color = (22, 61, 41) if cell["passed"] else (77, 28, 35)
    draw.rectangle((x, y, x + CELL_W - 1, y + LABEL_H - 1), fill=color)
    label = f"{page_number * 81 + slot + 1:02d} {cell['profile']}"
    if layout == "trials":
      label += f" #{cell['trial']}"
    draw.text((x + 5, y + 2), label, font=small, fill="white")
  legend = [
    ["V18 / model_2399", "9 x 9 test matrix", f"Page {page_number + 1}/{pages}"],
    [
      "73 test profiles",
      "36 original + 37 in-motion",
      "First trial per case" if layout == "cases" else "All 440 repeated trials",
    ],
    [
      "20s / 25fps / normal speed",
      "All cells share the same time",
      "No successful-take selection",
    ],
    ["CMD: target body velocity", "ACT: measured body velocity", "x/y: m/s   w: rad/s"],
    [
      "Green header: full-profile PASS",
      "Red header: full-profile FAIL",
      "Includes speed, gait, lift, stop",
    ],
    [
      "Payload: central +3 kg",
      "Per-leg phase target: 1.5 Hz",
      "No heightscan / touchdown bonus",
    ],
    [
      "Original test seed: 8456",
      "In-motion test seed: 12456",
      "Recording replay, not acceptance",
    ],
    ["In-motion switches: 6/10/14s", "vx retained from 2s to 18s", "Final stop at 18s"],
  ]
  for slot in range(len(page), 81):
    x, y = slot % 9 * CELL_W, slot // 9 * CELL_H
    lines = legend[(slot - len(page)) % len(legend)]
    for j, line in enumerate(lines):
      draw.text((x + 18, y + 60 + j * 26), line, font=font, fill=(211, 221, 231))
    draw.text(
      (x + 18, y + 164),
      f"sha {plan['checkpoint_sha256'][:12]}",
      font=small,
      fill=(132, 152, 171),
    )
  return layer


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--layout", choices=("cases", "trials"), default="cases")
  parser.add_argument("--preview-frames", type=int, default=0)
  args = parser.parse_args()
  if args.output.exists():
    parser.error("Preserve earlier renders")
  assert (args.input_dir / "capture_complete.json").exists()
  plan = json.loads((args.input_dir / "recording_plan.json").read_text())
  egl_index = egl_gpu3()
  suites = [load_suite(args.input_dir, name) for name in ("original", "in_motion")]
  model = suites[0]["model"]
  other = suites[1]["model"]
  for field in ("mesh_vert", "mesh_face", "mesh_normal", "tex_data", "mat_texid"):
    assert np.array_equal(getattr(model, field), getattr(other, field)), field
  assert (model.nq, model.nv, model.ngeom) == (other.nq, other.nv, other.ngeom)
  disable_model_sameframe_shortcuts(model)
  model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
  model.vis.quality.offsamples = 0
  model.stat.extent = 4.0
  model.light_castshadow[:] = False
  model.mat_reflectance[:] = 0
  model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
  model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
  context = mujoco.GLContext(WIDTH, HEIGHT)
  context.make_current()
  render_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
  mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, render_context)
  scene = mujoco.MjvScene(model, maxgeom=10000)
  scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
  scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
  scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
  option = mujoco.MjvOption()
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.azimuth, camera.elevation, camera.distance = 130, -24, 1.05
  data = mujoco.MjData(model)
  pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
  rect = mujoco.MjrRect(0, 0, WIDTH, HEIGHT)
  font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
  small = ImageFont.truetype(font_path, 13)
  font = ImageFont.truetype(font_path, 17)
  cells = cells_for(suites, args.layout)
  pages = math.ceil(len(cells) / 81)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  render_plan = {
    "protocol": "pochi-native-9x9-video-v1",
    "layout": args.layout,
    "cells": len(cells),
    "pages": pages,
    "width": WIDTH,
    "height": HEIGHT,
    "fps": FPS,
    "physical_gpu": 3,
    "egl_device_index": egl_index,
    "source_sha256": digest(__file__),
    "checkpoint_sha256": plan["checkpoint_sha256"],
    "normal_speed": True,
    "first_trial_only": args.layout == "cases",
    "new_selection": False,
  }
  args.output.with_suffix(".plan.json").write_text(json.dumps(render_plan, indent=2))
  encoder = subprocess.Popen(
    [
      "ffmpeg",
      "-hide_banner",
      "-loglevel",
      "warning",
      "-nostdin",
      "-n",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgb24",
      "-s",
      f"{WIDTH}x{HEIGHT}",
      "-r",
      str(FPS),
      "-i",
      "pipe:0",
      "-an",
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-crf",
      "18",
      "-threads",
      "8",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      str(args.output),
    ],
    stdin=subprocess.PIPE,
  )
  started = time.monotonic()
  rendered = 0
  index_rows = []
  try:
    for page_number in range(pages):
      page = cells[page_number * 81 : (page_number + 1) * 81]
      labels = make_labels(page, page_number, pages, plan, font, small, args.layout)
      for slot, cell in enumerate(page):
        index_rows.append(
          {
            "page": page_number + 1,
            "row": slot // 9 + 1,
            "column": slot % 9 + 1,
            "profile": cell["profile"],
            "suite": cell["suite"]["name"],
            "trial": cell["trial"],
            "passed": cell["passed"],
            "commands": cell["segments"],
          }
        )
      frames = min(500, args.preview_frames) if args.preview_frames else 500
      for frame in range(frames):
        step = frame * 2 + 1
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glClearColor(0.047, 0.059, 0.078, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        for slot, cell in enumerate(page):
          states, fields = cell["suite"]["states"], cell["suite"]["fields"]
          env_id = cell["index"]
          invalid = np.flatnonzero(~states["valid"][:, env_id])
          pose_step = min(step, max(int(invalid[0]) - 1, 0)) if len(invalid) else step
          for name, value in fields.items():
            target = getattr(model, name)
            target[:] = value[env_id].reshape(target.shape)
          data.qpos[:] = states["qpos"][pose_step, env_id]
          data.qvel[:] = states["qvel"][pose_step, env_id]
          mujoco.mj_kinematics(model, data)
          mujoco.mj_comPos(model, data)
          mujoco.mj_camlight(model, data)
          camera.lookat[:] = data.qpos[:3] + [0, 0, -0.12]
          mujoco.mjv_updateScene(
            model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
          )
          view = mujoco.MjrRect(
            slot % 9 * CELL_W,
            HEIGHT - (slot // 9 + 1) * CELL_H,
            CELL_W,
            CELL_H - LABEL_H,
          )
          mujoco.mjr_render(view, scene, render_context)
        mujoco.mjr_readPixels(pixels, None, rect, render_context)
        image = Image.fromarray(pixels[::-1])
        draw = ImageDraw.Draw(image)
        for slot, cell in enumerate(page):
          x, y = slot % 9 * CELL_W, slot // 9 * CELL_H
          image.paste(labels.crop((x, y, x + CELL_W, y + LABEL_H)), (x, y))
          command = command_at(cell["segments"], step * 0.02)
          actual = cell["suite"]["states"]["velocity"][step, cell["index"]]
          draw.text(
            (x + 5, y + 19),
            vector_line("CMD", command),
            font=small,
            fill=(193, 221, 252),
          )
          draw.text(
            (x + 5, y + 36),
            vector_line("ACT", actual),
            font=small,
            fill=(235, 235, 235),
          )
          draw.rectangle((x, y, x + CELL_W - 1, y + CELL_H - 1), outline=(86, 91, 103))
        for slot in range(len(page), 81):
          x, y = slot % 9 * CELL_W, slot // 9 * CELL_H
          image.paste(labels.crop((x, y, x + CELL_W, y + CELL_H)), (x, y))
        if len(page) < 81:
          slot = len(page)
          draw.text(
            (slot % 9 * CELL_W + 18, slot // 9 * CELL_H + 194),
            f"t = {(step + 1) * 0.02:05.2f} / 20.00 s",
            font=font,
            fill=(255, 219, 121),
          )
        if frame == min(274, frames - 1) and page_number == 0:
          image.save(args.output.with_suffix(".jpg"), quality=93)
        encoder.stdin.write(image.tobytes())
        rendered += 1
        if frame % 25 == 0:
          print(
            f"Page {page_number + 1}/{pages} frame {frame + 1}/{frames} "
            f"elapsed {time.monotonic() - started:.1f}s",
            flush=True,
          )
    encoder.stdin.close()
    if encoder.wait() != 0:
      raise RuntimeError("Video encoding failed")
  finally:
    if encoder.poll() is None:
      encoder.stdin.close()
      encoder.wait()
    render_context.free()
    context.free()
  result = {
    **render_plan,
    "frames": rendered,
    "duration_s": rendered / FPS,
    "video": str(args.output),
    "video_sha256": digest(args.output),
    "cells_index": index_rows,
    "completed": True,
  }
  args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
  print(
    json.dumps(
      {"video": str(args.output), "frames": rendered, "duration_s": rendered / FPS}
    ),
    flush=True,
  )


if __name__ == "__main__":
  main()
