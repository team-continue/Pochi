# Pochi Hardware

Pochiの回路、STM32/Teensyファームウェア、ホスト側通信ツール、Python API、
Web joint viewerをまとめています。

## Python APIとWeb joint viewer

`pochi_client`はTeensy USB/UDP bridgeと通信するPython APIです。12台分のMIT指令を
1 packetで送信し、12台の状態とIMUを1 packetで受信します。Web viewerは左側の
3Dモデルへ現在姿勢を反映し、右側から関節の選択・目標角度・全体Torque ON/OFFを
操作できます。

### セットアップ

```bash
cd ~/Documents/GitHub/Pochi
uv sync

cd pochi_hardware/web
npm install
```

### Web joint viewer

3つのターミナルを使います。

#### 1. USB/UDP bridge

macOSではTeensyのdevice名を確認して起動します。

```bash
cd ~/Documents/GitHub/Pochi
uv run python pochi_hardware/src/teensy_udp_bridge.py \
  --serial /dev/cu.usbmodem176277501 \
  --command-bind 127.0.0.1:15000 \
  --state-dest 127.0.0.1:15001
```

#### 2. UDP/WebSocket server

```bash
cd ~/Documents/GitHub/Pochi
uv run python -m pochi_hardware.web.server
```

状態確認用URLは`http://127.0.0.1:8765/health`、WebSocketは
`ws://127.0.0.1:8765/ws`です。

#### 3. Web UI

```bash
cd ~/Documents/GitHub/Pochi/pochi_hardware/web
npm run dev
```

ブラウザで`http://127.0.0.1:3000`を開きます。

### Viewer controls and safety

- 左の3Dモデルは12軸の実角度を常時反映します。ドラッグで視点回転、スクロールでズーム、関節をクリックすると選択できます。
- 右の関節一覧または3Dモデルから1軸を選び、スライダーでMIT controlの目標角を更新します。
- `TORQUE ON`は全12台の現在角を初期目標に設定してから全軸をEnableします。
- 全12台から有効なCAN feedbackが返り、Faultがないときだけ`TORQUE ON`できます。
- 最後のWeb画面を閉じた場合、WebSocket serverは全軸を自動Disableします。
- WebSocket server終了時は全軸Disableとemergency-stop packetを送ります。
- Hip/Thigh/Calfの暫定UI制限はそれぞれ±55°、±150°、-165〜25°です。機構限界確定後にWeb UIとserverの両方を更新してください。

### Client API

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

### Joint layout

現時点の脚とCAN IDの対応は仮配置です。

| Leg | Hip | Thigh | Calf |
| --- | ---: | ---: | ---: |
| Front left | 1 | 2 | 3 |
| Front right | 4 | 5 | 6 |
| Rear left | 7 | 8 | 9 |
| Rear right | 10 | 11 | 12 |

実機配線が確定したら`pochi_hardware/client/pochi_client/joint_layout.py`を変更します。
Web画面には誤認防止のためCAN IDを常時表示します。

### テストとWebビルド

```bash
cd ~/Documents/GitHub/Pochi
uv run pytest -q

cd pochi_hardware/web
npm run build
```

## Jetson Orin NanoのSPI0ピン設定

`systemd/pochi-spi0-pinmux.service`は、Jetson Orin Nanoの40ピンヘッダーにある
SPI0をPochi用に設定します。

使用する物理ピンは次の4本です。

| 物理ピン | 信号 |
| --- | --- |
| 19 | SPI0 MOSI |
| 21 | SPI0 MISO |
| 23 | SPI0 SCK |
| 24 | SPI0 CS0 |

### インストールと有効化

以下は、この`pochi_hardware`ディレクトリで実行します。

```bash
sudo install -o root -g root -m 0644 \
  systemd/pochi-spi0-pinmux.service \
  /etc/systemd/system/pochi-spi0-pinmux.service

sudo systemctl daemon-reload
sudo systemctl enable --now pochi-spi0-pinmux.service
sudo systemctl restart pochi-spi0-pinmux.service
```

状態を確認します。

```bash
systemctl is-enabled pochi-spi0-pinmux.service
systemctl is-active pochi-spi0-pinmux.service
```

それぞれ`enabled`と`active`になれば設定完了です。

### STM32との通信確認

STM32へSPIエコーファームウェアを書き込んだ状態で、Orinから20 MHzのテストを
実行します。

```bash
cd ~/pochi
./orin_spi_test /dev/spidev0.0 20000000
```

成功時は次のように表示されます。

```text
TX=DE AD BE E1  ECHO=DE AD BE E1
PASS: STM32 echoed the Orin request
```

### 停止とGPIO設定の復元

サービスを停止すると`ExecStop`が実行され、SPIパッドを開発用Orinで取得した
元のGPIO設定へ戻します。

```bash
sudo systemctl disable --now pochi-spi0-pinmux.service
```

サービス自体も削除する場合は、停止後に次を実行します。

```bash
sudo rm -f /etc/systemd/system/pochi-spi0-pinmux.service
sudo systemctl daemon-reload
```
