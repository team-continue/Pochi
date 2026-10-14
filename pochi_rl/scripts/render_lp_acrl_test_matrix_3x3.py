"""Render nine representative V18 cases from the existing 9x9 recordings."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import render_lp_acrl_test_matrix as native
from OpenGL import GL
from PIL import Image, ImageDraw, ImageFont

mujoco, np = native.mujoco, native.np
WIDTH, HEIGHT, BANNER = 1920, 1080, 48
CELL_W, CELL_H, LABEL_H = 640, 344, 66
FPS = 25
SELECTION = (
  ("original", "stand", "静止"),
  ("original", "x_+0.5", "前進 0.50 m/s"),
  ("original", "x_-0.5", "後退 0.50 m/s"),
  ("original", "y_+0.3", "左移動 0.30 m/s"),
  ("original", "y_-0.3", "右移動 0.30 m/s"),
  ("original", "stop_start_yaw", "その場旋回：左 → 停止 → 右"),
  ("in_motion", "forward_0.25_lateral_right_first", "前進しながら左右切替"),
  ("in_motion", "forward_0.25_yaw_left_first", "前進しながら旋回切替"),
  ("in_motion", "forward_0.25_co_turn_left_first", "前進＋横移動＋旋回"),
)


def select_cells(suites):
  cells = []
  for suite_name, name, title in SELECTION:
    suite = suites[suite_name]
    index = suite["report"]["names"].index(name)
    assert suite["report"]["episodes"][index]["environment_index"] == index
    cells.append(
      {
        "suite": suite,
        "index": index,
        "profile": name,
        "title": title,
        "segments": suite["report"]["profiles"][name],
        "passed": suite["report"]["episodes"][index]["passed"],
      }
    )
  return cells


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists() or args.output.with_suffix(".plan.json").exists():
    parser.error("Preserve earlier renders")
  assert (args.input_dir / "capture_complete.json").exists()
  recording = json.loads((args.input_dir / "recording_plan.json").read_text())
  egl_index = native.egl_gpu3()
  suites = {
    name: native.load_suite(args.input_dir, name) for name in ("original", "in_motion")
  }
  cells = select_cells(suites)
  model, other = suites["original"]["model"], suites["in_motion"]["model"]
  for field in ("mesh_vert", "mesh_face", "mesh_normal", "tex_data", "mat_texid"):
    assert np.array_equal(getattr(model, field), getattr(other, field)), field
  assert (model.nq, model.nv, model.ngeom) == (other.nq, other.nv, other.ngeom)
  native.disable_model_sameframe_shortcuts(model)
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
  for flag in (
    mujoco.mjtRndFlag.mjRND_SHADOW,
    mujoco.mjtRndFlag.mjRND_REFLECTION,
    mujoco.mjtRndFlag.mjRND_FOG,
  ):
    scene.flags[flag] = False
  option, camera = mujoco.MjvOption(), mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.azimuth, camera.elevation, camera.distance = 130, -24, 1.05
  data = mujoco.MjData(model)
  pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
  rect = mujoco.MjrRect(0, 0, WIDTH, HEIGHT)
  japanese = ImageFont.truetype(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20
  )
  note = ImageFont.truetype(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16
  )
  mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 17)
  labels = Image.new("RGB", (WIDTH, HEIGHT), (12, 15, 20))
  draw = ImageDraw.Draw(labels)
  draw.text((14, 0), "Pochi v18 / 代表9テスト・3×3", font=japanese, fill="white")
  draw.text((630, 3), "緑：総合合格", font=note, fill=(134, 221, 167))
  draw.text((780, 3), "赤：総合条件の未達", font=note, fill=(255, 160, 166))
  draw.text(
    (14, 25),
    "CMD＝指令 / ACT＝実測　x・y：m/s、w：rad/s　各種類の最初の1試行を同期再生",
    font=note,
    fill=(181, 195, 211),
  )
  index_rows = []
  for slot, cell in enumerate(cells):
    x, y = slot % 3 * CELL_W, BANNER + slot // 3 * CELL_H
    color = (22, 61, 41) if cell["passed"] else (77, 28, 35)
    draw.rectangle((x, y, x + CELL_W - 1, y + LABEL_H - 1), fill=color)
    draw.text((x + 9, y), f"{slot + 1}. {cell['title']}", font=japanese, fill="white")
    index_rows.append(
      {
        "row": slot // 3 + 1,
        "column": slot % 3 + 1,
        "title": cell["title"],
        "suite": cell["suite"]["name"],
        "profile": cell["profile"],
        "environment_index": cell["index"],
        "trial": 1,
        "passed": cell["passed"],
        "commands": cell["segments"],
      }
    )
  plan = {
    "protocol": "pochi-representative-3x3-video-v1",
    "rows": 3,
    "columns": 3,
    "width": WIDTH,
    "height": HEIGHT,
    "fps": FPS,
    "duration_s": 20,
    "same_recorded_trajectories_as_9x9": True,
    "no_new_simulations": True,
    "source_sha256": native.digest(__file__),
    "native_renderer_source_sha256": native.digest(native.__file__),
    "physical_gpu": 3,
    "egl_device_index": egl_index,
    "checkpoint_sha256": recording["checkpoint_sha256"],
    "selection": index_rows,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.with_suffix(".plan.json").write_text(
    json.dumps(plan, indent=2, ensure_ascii=False)
  )
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
      "6",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      str(args.output),
    ],
    stdin=subprocess.PIPE,
  )
  started = time.monotonic()
  try:
    for frame in range(500):
      step = frame * 2 + 1
      GL.glDisable(GL.GL_SCISSOR_TEST)
      GL.glClearColor(0.047, 0.059, 0.078, 1.0)
      GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
      for slot, cell in enumerate(cells):
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
          slot % 3 * CELL_W,
          HEIGHT - BANNER - (slot // 3 + 1) * CELL_H,
          CELL_W,
          CELL_H - LABEL_H,
        )
        mujoco.mjr_render(view, scene, render_context)
      mujoco.mjr_readPixels(pixels, None, rect, render_context)
      image = Image.fromarray(pixels[::-1])
      image.paste(labels.crop((0, 0, WIDTH, BANNER)), (0, 0))
      draw = ImageDraw.Draw(image)
      draw.text(
        (1630, 10),
        f"t = {(step + 1) * 0.02:05.2f} / 20s",
        font=mono,
        fill=(255, 219, 121),
      )
      for slot, cell in enumerate(cells):
        x, y = slot % 3 * CELL_W, BANNER + slot // 3 * CELL_H
        image.paste(labels.crop((x, y, x + CELL_W, y + LABEL_H)), (x, y))
        command = native.command_at(cell["segments"], step * 0.02)
        actual = cell["suite"]["states"]["velocity"][step, cell["index"]]
        draw.text(
          (x + 9, y + 27),
          native.vector_line("CMD", command),
          font=mono,
          fill=(193, 221, 252),
        )
        draw.text(
          (x + 9, y + 46),
          native.vector_line("ACT", actual),
          font=mono,
          fill=(235, 235, 235),
        )
        draw.rectangle((x, y, x + CELL_W - 1, y + CELL_H - 1), outline=(86, 91, 103))
      if frame == 274:
        image.save(args.output.with_suffix(".jpg"), quality=94)
      encoder.stdin.write(image.tobytes())
      if frame % 50 == 0:
        print(
          f"Frame {frame + 1}/500, elapsed {time.monotonic() - started:.1f}s",
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
  args.output.with_suffix(".json").write_text(
    json.dumps(
      {
        **plan,
        "completed": True,
        "frames": 500,
        "video": str(args.output),
        "video_sha256": native.digest(args.output),
      },
      indent=2,
      ensure_ascii=False,
    )
  )
  print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
  main()
