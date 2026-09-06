# Pochi RL

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

## Notes

- The MJCF uses primitive collision/visual geometry only. Keep collision
  primitives when CAD meshes arrive unless a task needs detailed contacts.
- Joint names and action order are defined in
  `src/pochi_rl/robot/pochi_constants.py`.
- The default control path is MuJoCo position actuators with `kp=40`,
  `forcerange="-30 30"`, and policy deltas scaled by `action_scale=0.25`.
- Masses, inertias, dimensions, joint limits, and the default crouch are starter
  values and should be replaced with CAD/hardware values before sim-to-real.

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
