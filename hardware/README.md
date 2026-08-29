# Pochi Hardware Client

`pochi_client`は、Teensy USB/UDP bridgeと通信するPython APIです。12台分のMIT指令を常に1 packetで送信し、12台の状態とIMUを1 packetで受信します。Web viewerは左側の3Dモデルへ現在姿勢を反映し、右側から関節の選択・目標角度・全体Torque ON/OFFを操作できます。

## Setup

```bash
cd ~/Documents/GitHub/Pochi
uv sync

cd hardware/web
npm install
```

## Web joint viewer

3つのターミナルを使います。

### 1. USB/UDP bridge

macOSではTeensyのdevice名を確認して起動します。

```bash
cd ~/Documents/GitHub/Pochi
uv run python pochi_hardware/src/teensy_udp_bridge.py \
  --serial /dev/cu.usbmodem176277501 \
  --command-bind 127.0.0.1:15000 \
  --state-dest 127.0.0.1:15001
```

### 2. UDP/WebSocket server

```bash
cd ~/Documents/GitHub/Pochi
uv run python -m hardware.web.server
```

状態確認用URLは`http://127.0.0.1:8765/health`、WebSocketは`ws://127.0.0.1:8765/ws`です。

### 3. Web UI

```bash
cd ~/Documents/GitHub/Pochi/hardware/web
npm run dev
```

ブラウザで`http://127.0.0.1:3000`を開きます。

## Viewer controls and safety

- 左の3Dモデルは12軸の実角度を常時反映します。ドラッグで視点回転、スクロールでズーム、関節をクリックすると選択できます。
- 右の関節一覧または3Dモデルから1軸を選び、スライダーでMIT controlの目標角を更新します。
- `TORQUE ON`は全12台の現在角を初期目標に設定してから全軸をEnableします。
- 全12台から有効なCAN feedbackが返り、Faultがないときだけ`TORQUE ON`できます。
- 最後のWeb画面を閉じた場合、WebSocket serverは全軸を自動Disableします。
- WebSocket server終了時は全軸Disableとemergency-stop packetを送ります。
- Hip/Thigh/Calfの暫定UI制限はそれぞれ±55°、±150°、-165〜25°です。機構限界確定後にWeb UIとserverの両方を更新してください。

## Client API

```python
from pochi_client import PochiClient

with PochiClient() as robot:
    state = robot.wait_for_state(timeout=1.0)
    robot.set_mit(
        "front_left_hip",
        position_rad=0.2,
        velocity_rad_s=0.0,
        kp=20.0,
        kd=0.5,
        torque_nm=0.0,
        enable=True,
    )
    robot.disable_all()
```

## Joint layout

現時点の脚とCAN IDの対応は仮配置です。

| Leg | Hip | Thigh | Calf |
| --- | ---: | ---: | ---: |
| Front left | 1 | 2 | 3 |
| Front right | 4 | 5 | 6 |
| Rear left | 7 | 8 | 9 |
| Rear right | 10 | 11 | 12 |

実機配線が確定したら`hardware/client/pochi_client/joint_layout.py`を変更します。Web画面には誤認防止のためCAN IDを常時表示します。

## Build and test

```bash
cd ~/Documents/GitHub/Pochi
uv run pytest -q

cd hardware/web
npm run build
```
