# Pochi Hardware

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
