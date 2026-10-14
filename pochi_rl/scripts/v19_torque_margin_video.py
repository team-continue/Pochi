"""Usual nine V19 cases as a 3x3 video, without comparative evaluation."""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tensordict import TensorDict
from v19_hardware_ablation import actor_from_saved, digest, write_json
from pochi_rl.mjlab.torque_margin import configuration as margin_configuration, safety_contract

def configuration(variant, num_envs, seed, *, play=False):
    return margin_configuration(num_envs, seed, play=play)

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
LABELS = {"torque_margin_8nm": "トルク余裕 8Nm", "noise": "観測ノイズ", "penalties": "振動抑制報酬", "noise_penalties": "ノイズ＋報酬",
    "noise_penalties_delay": "ノイズ＋報酬＋遅延", "baseline_v19": "元のv19"}


def command_at(segments, elapsed):
    for start, end, command in segments:
        if start <= elapsed < end:
            return command
    return segments[-1][2]


def visible_egl_device():
    from mujoco.egl import egl_ext as EGL
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_long))(EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
    candidates = []
    for index, device in enumerate(EGL.eglQueryDevicesEXT()):
        ordinal = ctypes.c_long(-1)
        if query(device, 0x323A, ctypes.byref(ordinal)) and ordinal.value == 0:
            candidates.append(index)
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot identify the sole visible CUDA device for EGL: {candidates}")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(candidates[0])
    return candidates[0]


@torch.inference_mode()
def capture(checkpoint, output, profiles):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS
    from pochi_rl.lp_acrl.physics import invalid_physics_mask
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend
    backend = install_contact_backend()
    configure_torch_backends()
    torch.set_num_threads(4)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = actor_from_saved(saved).to("cuda:0")
    cfg, agent = configuration("noise", 9, 8456, play=True)
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        assert abs(env.step_dt - .02) < 1e-9
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        term = env.command_manager.get_term("base_velocity")
        term.set_evaluation(task_ids=0, command=(0., 0., 0.))
        env.reset()
        robot = env.scene["robot"]
        fields = {name: getattr(env.sim.model, name).detach().cpu().numpy().copy()
            for name in env.sim.expanded_fields & VIEWER_MODEL_FIELDS}
        model_path = output.with_suffix(".mjb")
        mujoco.mj_saveModel(env.sim.mj_model, str(model_path))
        active = torch.ones(9, dtype=torch.bool, device="cuda:0")
        stopped_at, reason, frames = [None] * 9, [None] * 9, []
        for step in range(1000):
            command = torch.tensor([command_at(p, step * .02) for p in profiles], device="cuda:0")
            term.vel_command_b[:] = command
            term.vel_command_w[:] = command
            term.is_standing_env[:] = command.abs().amax(-1) <= 1e-6
            obs = TensorDict({name: env.observation_manager.compute_group(name, update_history=False)
                for name in env.cfg.observations}, batch_size=[9])
            obs["actor"][~active] = 0
            action = actor(obs)
            action[~active] = 0
            _, reward, done, _ = wrapped.step(action)
            valid = ~invalid_physics_mask(robot.data) & torch.isfinite(reward)
            qpos = env.sim.data.qpos.detach().cpu().numpy().copy()
            qvel = env.sim.data.qvel.detach().cpu().numpy().copy()
            velocity = torch.stack((robot.data.root_link_lin_vel_b[:, 0], robot.data.root_link_lin_vel_b[:, 1],
                robot.data.root_link_ang_vel_b[:, 2]), -1).cpu().numpy().copy()
            finite = np.isfinite(qpos).all(-1) & np.isfinite(qvel).all(-1)
            if frames:
                frozen = ~active.cpu().numpy() | ~finite
                for array, key in ((qpos, "qpos"), (qvel, "qvel"), (velocity, "velocity")):
                    array[frozen] = frames[-1][key][frozen]
            terminated = env.reset_terminated.bool()
            for index in (active & (terminated | ~valid)).nonzero().flatten().tolist():
                stopped_at[index] = (step + 1) * .02
                reason[index] = "転倒で停止" if bool(terminated[index]) else "物理状態異常で停止"
            frames.append({"qpos": qpos, "qvel": qvel, "velocity": velocity})
            finished = done.bool() | ~valid
            active &= ~finished
            if finished.any() and step < 999:
                env.reset(env_ids=finished.nonzero().flatten())
            if step % 250 == 0:
                print(f"Captured {step + 1}/1000", flush=True)
        states = {key: np.stack([f[key] for f in frames]) for key in frames[0]}
        np.savez_compressed(output.with_suffix(".states.npz"), **states)
        np.savez_compressed(output.with_suffix(".visual_fields.npz"), **fields)
        assert saved["infos"]["pochi_torque_margin_contract"] == safety_contract()
        state = saved["infos"]["pochi_v19_torque_margin_state"]
        return model_path, fields, states, stopped_at, reason, state, backend
    finally:
        env.close()


def render(output, model_path, fields, states, profiles, stops, reasons, state, egl_index, preview_frames):
    from OpenGL import GL
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    model = mujoco.MjModel.from_binary_path(str(model_path))
    disable_model_sameframe_shortcuts(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    model.vis.quality.offsamples = 0
    model.stat.extent = 4.
    model.light_castshadow[:] = False
    model.mat_reflectance[:] = 0
    model.vis.headlight.ambient[:] = [.4, .4, .4]
    model.vis.headlight.diffuse[:] = [.8, .8, .8]
    context = mujoco.GLContext(WIDTH, HEIGHT)
    context.make_current()
    rendering = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, rendering)
    scene = mujoco.MjvScene(model, maxgeom=10000)
    for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION, mujoco.mjtRndFlag.mjRND_FOG):
        scene.flags[flag] = False
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 130, -24, 1.05
    option, data = mujoco.MjvOption(), mujoco.MjData(model)
    pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rect = mujoco.MjrRect(0, 0, WIDTH, HEIGHT)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)
    small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
    mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 17)
    encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "4",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE)
    frame_count = preview_frames or 500
    started = time.monotonic()
    try:
        for frame in range(frame_count):
            step = frame * 2 + 1
            GL.glDisable(GL.GL_SCISSOR_TEST)
            GL.glClearColor(.047, .059, .078, 1.)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            for slot in range(9):
                for name, array in fields.items():
                    target = getattr(model, name)
                    target[:] = array[slot].reshape(target.shape)
                data.qpos[:], data.qvel[:] = states["qpos"][step, slot], states["qvel"][step, slot]
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                mujoco.mj_camlight(model, data)
                camera.lookat[:] = data.qpos[:3] + [0., 0., -.12]
                mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
                view = mujoco.MjrRect(slot % 3 * CELL_W, HEIGHT - BANNER - (slot // 3 + 1) * CELL_H,
                    CELL_W, CELL_H - LABEL_H)
                mujoco.mjr_render(view, scene, rendering)
            mujoco.mjr_readPixels(pixels, None, rect, rendering)
            image = Image.fromarray(pixels[::-1])
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, WIDTH, BANNER), fill=(12, 15, 20))
            draw.text((14, 0), f"Pochi / {LABELS[state['variant']]} / {state['completed_updates']}更新 / 3×3", font=font, fill="white")
            draw.text((14, 25), "CMD＝指令 / ACT＝実測　x・y：m/s、w：rad/s　同じ9ケース・通常速度・20秒", font=small, fill=(181, 195, 211))
            draw.text((1640, 10), f"t = {(step+1)*.02:05.2f} / 20s", font=mono, fill=(255, 219, 121))
            for slot, (_, _, title) in enumerate(SELECTION):
                x, y = slot % 3 * CELL_W, BANNER + slot // 3 * CELL_H
                stopped = stops[slot] is not None and (step + 1) * .02 >= stops[slot]
                draw.rectangle((x, y, x + CELL_W - 1, y + LABEL_H - 1), fill=(77, 28, 35) if stopped else (24, 40, 61))
                draw.text((x + 9, y), f"{slot+1}. {title}", font=font, fill="white")
                for line, prefix, values, color in ((27, "CMD", command_at(profiles[slot], step * .02), (193, 221, 252)),
                    (46, "ACT", states["velocity"][step, slot], (235, 235, 235))):
                    draw.text((x + 9, y + line), f"{prefix} x{values[0]:+.2f} y{values[1]:+.2f} w{values[2]:+.2f}", font=mono, fill=color)
                if stopped:
                    draw.text((x + 370, y + 46), reasons[slot], font=small, fill=(255, 190, 190))
                draw.rectangle((x, y, x + CELL_W - 1, y + CELL_H - 1), outline=(86, 91, 103))
            if frame == min(274, frame_count - 1):
                image.save(output.with_suffix(".jpg"), quality=94)
            encoder.stdin.write(image.tobytes())
            if frame % 50 == 0:
                print(f"Rendered {frame+1}/{frame_count}, {time.monotonic()-started:.1f}s, EGL {egl_index}", flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
    finally:
        if encoder.poll() is None:
            encoder.stdin.close()
            encoder.wait()
        rendering.free()
        context.free()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gpu", required=True, type=int, choices=(0,))
    p.add_argument("--preview-frames", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        p.error("Preserve earlier videos")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from eval_lp_acrl_omni_phase import PROFILE_SEGMENTS
    from eval_lp_acrl_omni_transitions import profiles as transition_profiles
    suites = {"original": PROFILE_SEGMENTS, "in_motion": transition_profiles()}
    profiles = [tuple((start, end, tuple(float(np.clip(value, -limit, limit))
        for value, limit in zip(command, (.5, .3, .6))))
        for start, end, command in suites[suite][name]) for suite, name, _ in SELECTION]
    # Select EGL by CUDA visibility before any GL context is created.
    torch.cuda.init()
    egl_index = visible_egl_device()
    model, fields, states, stops, reasons, state, backend = capture(args.checkpoint, args.output, profiles)
    render(args.output, model, fields, states, profiles, stops, reasons, state, egl_index, args.preview_frames)
    write_json(args.output.with_suffix(".json"), {"protocol": "v19-torque-margin-3x3-video-v1",
        "video": str(args.output.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint), "variant": state["variant"],
        "completed_updates": state["completed_updates"], "seed": 8456, "rows": 3, "columns": 3,
        "width": WIDTH, "height": HEIGHT, "fps": FPS, "duration_s": (args.preview_frames or 500) / FPS,
        "physical_gpu": args.gpu, "egl_device_index": egl_index, "contact_backend": backend,
        "cases": [{"title": title, "profile": name, "commands": commands, "stopped_at_s": stop, "stop_reason": reason}
            for (_, name, title), commands, stop, reason in zip(SELECTION, profiles, stops, reasons)],
        "evaluation_performed": False, "safety_contract": safety_contract(), "recording": "First trial of all nine cases; deterministic mean policy, clean playback observations, no command delay; terminated cases visibly freeze after stopping."})


if __name__ == "__main__":
    main()
