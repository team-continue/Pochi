# Teensy 4.1 PlatformIO test

Teensy 4.1のPlatformIO環境とCAN3上のRoboStride 03を確認するためのプロジェクトです。

- CAN3: Classic CAN、1 Mbps
- RoboStride 03: CAN ID 0〜11
- マスターID: `0xFD`
- 制御方式: MIT control
- 起動時の既定状態: 全モーター停止
- USB Serial: 115200 bps
- USB packet: COBS + `0x00` delimiter + CRC-32
- LED: 12台のencoderが見えている間は点灯、不足時は点滅

起動時は12台すべてにStopを送り、MIT modeを設定したうえでencoderの読み取りを開始します。トルクはWebから明示的にONにするまで入りません。

## Joint reference pose

`src/can3.h` の `CAN3_INITIAL_POSITION_RAD` が、CAN ID `0..11` に対応する
負荷側absolute encoderの初期姿勢です。外部へ送る角度と外部から受けるMIT目標角度は、
`normalizeAngle(raw - initial)` により初期姿勢を `0 rad` とする `[-pi, pi]` の値です。
生encoder座標との相互変換は `src/robostride.h` 内で行います。

MIT位置目標は `src/can3.h` の `CAN3_MIN_POSITION_RAD` と
`CAN3_MAX_POSITION_RAD` でID別に制限されます。モーター取り付け方向による符号反転は
Webの3D描画だけに適用され、Mechanical positionとMIT目標値はTeensy座標のままです。

## Build

```bash
cd ~/Documents/GitHub/Pochi/pochi_hardware/firmware/teensy41_test
~/.platformio/penv/bin/pio run
```

## Upload

Teensy 4.1をUSBで接続して実行します。自動書き込みが始まらない場合は、基板上のProgramボタンを1回押します。

```bash
~/.platformio/penv/bin/pio run -t upload
```

## Change a RoboStride CAN ID

`test/change_id.cpp` is a one-shot ID changer for one RS03 using the private
29-bit CAN protocol. Edit `kOldMotorId` and `kNewMotorId`, temporarily use the
file as `src/main.cpp`, then build and upload it. Connect exactly one target
motor while changing an ID.

The program first scans every valid private-protocol device ID from `0x01`
through `0xFE` on CAN3 and prints
each responding ID. It then verifies the old ID, refuses to proceed if the new
ID already responds, disables the motor, changes the ID, and verifies both that
the new ID responds and the old ID no longer responds. A pass leaves the Teensy
LED on; a failure blinks it. After changing the ID, the program sends the Type
22 motor-data-save frame (`01 02 03 04 05 06 07 08`) to persist the setting.
Power-cycle the motor and scan again to verify persistence.

After changing all IDs, restore the normal `src/main.cpp` before uploading the
control firmware.

The normal firmware implements MIT-style operation control using RoboStride's
private extended-frame protocol (`run_mode = 0`). Do not switch the motor to the
separate standard-frame MIT communication protocol. If that protocol was
selected previously, switch the motor back to the private protocol and power
cycle it first.

## USB / UDP

The Teensy built-in LED stays off without valid host commands and blinks at
20 Hz while valid command packets are arriving from the Orin. This indicates
host communication only; it does not indicate that motor torque is enabled.

Motor control uses a latched safety state. At boot, after a host-command
timeout, or when any RoboStride response is stale for 250 ms, the Teensy stops
all 12 motors and returns to non-blocking initialization. All motors must stay
responsive for 500 ms before the state becomes READY. Recovery never restores
torque automatically: the Teensy must receive an all-disabled command followed
by a fresh all-enabled command from the web UI. Enabling starts from the live
joint pose.

The command packet carries an independent Enable bit for every motor. The web
UI can request all motors or any subset, and unselected actuators remain in
`Reset` while their encoders are still sampled. Selected motors are enabled in
CAN-ID order, one at a time. An unconfirmed actuator receives another Enable
request every 50 ms, up to 20 attempts. After it answers with motor mode `Run`,
its captured-position MIT command becomes active and the firmware waits 500 ms
before starting the next selected actuator. A missing confirmation after 20
attempts or a later return to `Reset` stops every motor and latches another
re-arm request. The web UI reports `ENABLE PENDING`, `RESET`, `CALIBRATION`, and
`RUN` from actuator feedback rather than from the outgoing command alone.

Immediately before each Enable sequence, the Teensy captures that actuator's
latest encoder position and sends it as an MIT goal. It sends the same goal
again immediately after every Enable request and holds it for the first 100 ms
after the `Run` reply. This prevents a target retained by the actuator from a
previous run from moving the joint during startup.

Joint targets are converted back to the raw MIT coordinate using the `2pi`
equivalent angle nearest to the latest raw encoder reading. A plain
`offset + joint` conversion can cross the absolute-encoder wrap and request a
position one full revolution away while still appearing numerically valid in
the RS03 `[-4pi, 4pi]` MIT range.

```bash
cd ~/Documents/GitHub/Pochi
uv sync
uv run python pochi_hardware/src/teensy_usb_test.py --serial /dev/ttyACM0
```

USBとUDPを中継する場合は次を実行します。UDP commandはport 15000で受信し、stateはport 15001へ送信します。

```bash
uv run python pochi_hardware/src/teensy_udp_bridge.py \
  --serial /dev/ttyACM0 \
  --command-bind 0.0.0.0:15000 \
  --state-dest 127.0.0.1:15001
```

## Protocol

- Python → Teensy: 12台分のMIT指令を1個の316-byte packetで送信
- Teensy → Python: 12台分の状態とIMUを1個の860-byte packetで返信
- USB上ではCOBS encode後に`0x00`を付加
- UDP上ではCOBSを外したpacketを1 datagramとして送受信
- 指令が50 ms途絶えると全モーターを停止
- CRC、sequence、motor ID、有限値を検査してから12台分を一括更新

IMUはcore2025 upperと同じICM-20948を使用し、SPI1のMOSI 26、MISO 39、SCK 27、CS 38へ接続します。

## Test

```bash
cd ~/Documents/GitHub/Pochi
uv run pytest -q
```

実機に対して全モーターDisableのままUSB/UDP往復を確認する場合は、ブリッジを起動した別ターミナルから実行します。

```bash
uv run python pochi_hardware/src/teensy_udp_test.py --duration 3
```
