# Teensy 4.1 PlatformIO test

Teensy 4.1のPlatformIO環境とCAN3上のRoboStride 03を確認するためのプロジェクトです。

- CAN3: Classic CAN、1 Mbps
- RoboStride 03: CAN ID 1〜12
- マスターID: `0xFD`
- 制御方式: MIT control
- 起動時の既定状態: 全モーター停止
- USB Serial: 115200 bps
- USB packet: COBS + `0x00` delimiter + CRC-32
- LED: 初期化中は消灯、初期化完了後は点灯、有効なUSB指令受信中は点滅

起動時に12台を順番に初期化します。未接続のモーターがあっても永久待機せず、初期化結果をUSB Serialへ表示してメインループへ進みます。

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

## USB / UDP

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
