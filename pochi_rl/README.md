# Pochi RL

論文 arXiv:2601.17428 に基づく新しい独立歩行タスクは
[LP_ACRL.md](LP_ACRL.md) を参照してください。既存の歩行ポリシーとは別の
学習設定・チェックポイントで、ランダム初期化から学習します。

Standalone uv-managed reinforcement-learning project for the Pochi quadruped.
Phase 1 targets mjlab with a starter 4-leg x 3-DoF MJCF and a flat-ground
velocity-tracking PPO task. Phase 2 can add Isaac Lab configs while reusing the
same `task_spec.py` and robot constants.

## Setup

```bash
cd /home/k_suzuki/fun_ws/Pochi/pochi_rl
uv sync
uv pip install -e .
```

## Inspect the Robot

```bash
uv run python scripts/view_mjcf.py
```

The viewer uses CAD-detail visual meshes with sharp plate edges and smooth
curved surfaces. Regenerate them with `uv run python scripts/glb_to_mjcf.py
--report` after installing the CAD dependencies (`uv sync --group cad`).
`--visual-quality compact` opts into lighter, less detailed meshes. See
[model conventions](assets/pochi/README.md) for the imported archive and the
joint definitions used to assemble it.

## Train

Single GPU:

```bash
MUJOCO_GL=egl uv run train Pochi-Velocity-Flat-v0 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 1500 \
  --agent.experiment-name pochi_velocity_flat
```

Multi GPU:

```bash
MUJOCO_GL=egl uv run train Pochi-Velocity-Flat-v0 \
  --env.scene.num-envs 4096 \
  --gpu-ids "[0,1]"
```

Playback:

```bash
uv run play Pochi-Velocity-Flat-v0 \
  --checkpoint-file logs/rsl_rl/pochi_velocity_flat/<ts>/model_1500.pt \
  --num-envs 1
```

Export:

```bash
uv run python scripts/export_policy.py \
  --checkpoint logs/rsl_rl/pochi_velocity_flat/<ts>/model_1500.pt \
  --output exports/
```

## Deliberate, one-foot-at-a-time walking

`Pochi-Deliberate-Walk-v0` trains a separate low-speed policy. Its commands
cover 0.06–0.14 m/s forward, up to 0.025 m/s sideways and 0.15 rad/s turning.
The target gait is RL → FL → RR → FR: a 2.8 s cycle, 0.56 s per swing and
0.14 s with all four feet down between swings. The swing reference lifts the
foot 4.5 cm with zero vertical velocity at either end. Contact timing, foot
height, support and slipping are explicit objectives; these are learned
preferences, not hard guarantees on the resulting motion.

This task's training and playback use the hardware's RS02 MIT-mode gains:
**Kp = 140 N.m/rad, Kd = 1.5 N.m.s/rad**, set in `DeliberateWalkSpec`.
The position target is still `default_joint_pos + 0.25 * action`, updated at
50 Hz. The simulator applies joint-space PD torque at each 0.005 s physics
step with the existing torque-speed limits. The standalone MJCF, legacy
velocity task and stand-up task keep their own gains.

```bash
MUJOCO_GL=egl uv run train Pochi-Deliberate-Walk-v0 \
  --gpu-ids "[1]" --env.scene.num-envs 4096 --agent.max-iterations 1500

MUJOCO_GL=egl uv run play Pochi-Deliberate-Walk-v0 \
  --checkpoint-file logs/rsl_rl/pochi_deliberate_walk/<run>/model_1499.pt \
  --num-envs 1 --viewer viser

MUJOCO_GL=egl uv run python scripts/eval_deliberate_walk.py \
  --checkpoint logs/rsl_rl/pochi_deliberate_walk/<run>/model_1499.pt \
  --output /tmp/deliberate-walk-eval.json
```

Playback starts at 0.10 m/s straight ahead. The viewer has **Stop** and
**Walk slowly** buttons and centimetre-per-second velocity controls. It reports
its actual HTTP port on startup (8080, or the next free port).

The policy observes a sine/cosine gait clock in addition to the velocity
task's 48 observations, so it needs its own 50-input checkpoint. The clock is
per-episode time in seconds; it must be supplied consistently at deployment.
The phase, lift and contact schedule are disabled at near-zero commands.

The evaluator uses the same seed and commands for both tasks. Pass
`--task Pochi-Velocity-Flat-v0` with an older checkpoint for comparison. It
reports achieved velocity, joint speeds, support fraction, contact slip,
footfalls after at least 0.10 s airborne, and falls. A motionless or sliding
policy should not be accepted merely because its joint speeds are low.

Validated MIT-gain checkpoint (2026-09-05):
`logs/rsl_rl/pochi_deliberate_walk/2026-09-05_15-27-20_mit_kp140_kd1p5/model_1349.pt`.
This adapts `model_950.pt` for 400 additional PPO iterations at **Kp 140 / Kd 1.5**.
At a 0.10 m/s command, the old weights at Kp 140 achieved 0.077 m/s with 90.6%
three-foot-or-better support and 1.77 footfalls/s. After adaptation these were
0.102 m/s, 95.4% and 1.44 footfalls/s. A 32-environment continuous sequence
(0.10 → 0 → 0.10 → 0.14 → 0.06 m/s, 154 s total) had no falls; a separate
32-environment seed-123 evaluation of walking and stopping also had no falls.
At the 0.14 m/s command, actual speed was 0.124 m/s and support fraction was
92.5%; this is a measured operating limit, not perfect command tracking.
These are simulator results. Local reports and video are in
`logs/deliberate_walk_mit/final_continuous.json`,
`logs/deliberate_walk_mit/final_seed123.json` and
`logs/deliberate_walk_mit/walk_kp140.mp4`.

Historical checkpoint, trained and evaluated at **Kp 60 / Kd 1.5** (2026-09-05):
`logs/rsl_rl/pochi_deliberate_walk/2026-09-05_12-03-42_sequential/model_950.pt`.
It passed a 32-environment continuous sequence of 0.10 → 0 → 0.10 → 0.14 →
0.06 m/s, 30.8 s per command including settling, with no falls. At the default
0.10 m/s command it achieved 0.098 m/s, 1.44 total footfalls/s and 97% of time
with at least three feet in contact. Contact-site slip remained about 0.022 m/s;
contact timing and foot clearance are learned approximately. See
`logs/deliberate_walk/final_continuous.json` and
`logs/deliberate_walk/deliberate_walk.mp4` for the local results. Training was
stopped after this checkpoint passed evaluation; later intermediate checkpoints
are retained, but were not selected on these tests.

To reproduce an old checkpoint's gain in the evaluator, pass `--kp 60 --kd 1.5`.
The JSON report records the evaluation gains, seed and continuous/reset mode.
Use `--seed` for a separate randomization draw. Changing the task gains does
not update weights in an already downloaded `.pt` file.

For adaptation from the historical low-speed checkpoint:

```bash
MUJOCO_GL=egl uv run train Pochi-Deliberate-Walk-v0 \
  --gpu-ids "[1]" --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run 2026-09-05_12-03-42_sequential \
  --agent.load-checkpoint model_950.pt --agent.max-iterations 400 \
  --agent.run-name mit_kp140_kd1p5
```

The iteration count above is **additional** iterations after loading the model.
The new run's `params/env.yaml` records the gains actually used for learning.

## Notes

- The MJCF uses CAD visual meshes, primitive body/limb collisions and CAD foot
  pad convex hulls. Visual quality settings do not change those collisions.
- Joint names and action order are defined in
  `src/pochi_rl/robot/pochi_constants.py`.
- In mjlab, explicit PD actuators convert policy position targets to torque.
  Deliberate walking uses `kp=140`, `kd=1.5` and `action_scale=0.25`.
- Masses, inertias, dimensions, joint limits, and the default crouch are starter
  values and should be replaced with CAD/hardware values before sim-to-real.

## MPC Walking (No Training)

```bash
uv run --extra mpc python scripts/play_mpc_walk.py --port 8092
```

Open `http://localhost:8092` for live MuJoCo walking with MPC trajectory and
contact-force overlays. See [MPC_WALK.md](MPC_WALK.md) for evaluation, controls,
and the controller's scope.


## IK 横ジャンプ（学習不要）

3次元の足先IKで沈み込みと踏み切りを作り、全脚の離地を検知して空中姿勢へ
切り替えます。接地後は関節PDと胴体の位置・姿勢フィードバックで衝撃を吸収し、
まず四脚で減速します。その後、三脚で支えながら一脚ずつ踏み直して標準の足位置に
戻し、胴体を水平にして終了します。IKだけでは跳躍の力を決められないため、支持脚への力配分を
併用しています。浮遊胴体を直接操作せず、RS02のトルク・速度制限を適用しています。

```bash
# ブラウザで再生（表示された localhost URL を開く）
uv run --extra mpc python scripts/play_side_jump.py --direction left --port 8093

# 描画なしの評価。着地して静止できなければ終了コード1
uv run python scripts/eval_side_jump.py --direction left
uv run python scripts/eval_side_jump.py --direction right
```

`left` はロボットの +Y、`right` は −Y 方向です。ビューアで大きさと方向を選択後、
**Controls → Reset** で再実行、**Slower** でスロー再生できます。
`--extra mpc` は既存のビューア依存パッケージを利用するための指定で、
この制御はMPCや学習済みモデルを使いません。

標準は **大ジャンプ** です。深い沈み込み（22 cm）と強い上向きの踏み切りを
横速度の増加と組み合わせ、着地できる滞空時間を確保しています。

| `--jump-size` | 横移動（準備から静止まで） | 空中での横移動 | 全脚の離地時間 |
| --- | ---: | ---: | ---: |
| `small` | 約13 cm | 約6 cm | 約0.20秒 |
| `medium` | 約33 cm | 約20 cm | 約0.36秒 |
| `large`（標準） | 約52 cm | 約31 cm | 約0.34秒 |

```bash
uv run python scripts/eval_side_jump.py --jump-size large --direction right
uv run --extra mpc python scripts/play_side_jump.py --jump-size small
```

値は標準の床・機体での実測目安で、指定距離に着地する制御ではありません。
大ジャンプの最大傾きは約11度、胴体の最高位置は約49 cmです。
`--lateral-speed` は選択したプリセットの踏み切り横速度目標を上書きします
（最大2.4 m/s）。実際の離地速度や移動距離とは異なり、プリセット外の組み合わせで
同じ着地性能を保証しません。左右の全プリセットに対して、空中移動、胴体非接触、
四脚着地、静止と各ステップのモータートルク・速度制限を回帰テストしています。
着地後の踏み直しには約6秒かかり、全体は約10秒で評価します。終了判定には、
四脚接地・静止に加え、胴体の姿勢誤差1度未満（ヨーを含む）と標準関節角との
最大差1.5度未満が0.5秒続くことを要求します。着地時の踏ん張った姿勢では
完了にしません。評価JSONとビューアに現在の姿勢誤差を表示します。
平坦な床でのシミュレーション用デモで、実機の着地性能は未検証です。

実装: `src/pochi_rl/control/side_jump.py`、3次元IK: `control/leg_kinematics.py`。
