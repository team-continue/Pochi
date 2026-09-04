# Hardware reference documents

2026-08-23 に `main_1` 回路図を照合するため収集した資料です。
可能なものはメーカー公式PDFを保存し、メーカーサイトが直接取得を拒否する場合のみ、同一メーカー文書の配布ミラーを利用しています。

| 対象 | ローカル資料 | 公式の最新版・製品ページ | 備考 |
|---|---|---|---|
| STM32G474CB | `ST_STM32G474_datasheet.pdf` | https://www.st.com/resource/en/datasheet/stm32g474cb.pdf | DS12288 Rev.6 |
| STM32G4 hardware guide | 未保存 | https://www.st.com/resource/en/application_note/an5093-getting-started-with-stm32g4-series--hardware-development-boards-stmicroelectronics.pdf | AN5093 Rev.2。STサーバーがCLI取得を拒否するためリンクのみ |
| ICM-45686 | `TDK_ICM-45686_datasheet.pdf` | https://invensense.tdk.com/wp-content/uploads/documentation/DS-000577_ICM-45686.pdf | DS-000577 Rev.1.0。配布ミラーから保存 |
| S9012 / C8543 | `JSCJ_S9012_C8543_datasheet.pdf` | https://www.jscj-elec.com/gallery/file/AD-S9012%20series.pdf | メーカー公式PDF。C8543はSOT-23、hFE 200–350品 |
| SMCJ43A | `Bourns_SMCJ_datasheet.pdf` | https://www.bourns.com/docs/Product-Datasheets/SMCJ.pdf | 同一電気仕様のBourns公式シリーズ資料。採用時はメーカーとMPNを固定する |
| DDZ9699 12 V Zener | `Diodes_DDZ9699_datasheet.pdf` | https://www.diodes.com/part/view/DDZ9699 | LTC4359のGATE–SOURCE保護候補、SOD-123 |
| RobStride discharge module | `RobStride_discharge_module_manual_V1.1.pdf` | https://github.com/RobStride/Product_Information/tree/main/%E4%BA%A7%E5%93%81%E8%B5%84%E6%96%99/dischargeModule | 15–60 V、48 Vモードは53.5 Vで泄放開始。3 Ω/500 W～10 Ω/200 Wを推奨 |
| RobStride power board | `RobStride_power_board_manual_V1.5.pdf` | https://github.com/RobStride/Product_Information/tree/main/%E4%BA%A7%E5%93%81%E8%B5%84%E6%96%99/powerBoard | 20–65 V、4支路、プリチャージ・泄放・過流保護の構成参考 |
| TPS5430 | `TI_TPS5430_datasheet.pdf` | https://www.ti.com/lit/ds/symlink/tps5430.pdf | 公式PDF |
| ISO1044 | `TI_ISO1044_datasheet.pdf` | https://www.ti.com/lit/ds/symlink/iso1044.pdf | 公式PDF |
| ADuM120N/ADuM121N | `ADI_ADuM120N_ADuM121N_datasheet_RevE.pdf` | https://www.analog.com/media/en/technical-documentation/data-sheets/ADuM120N_121N.pdf | ローカルはRev.E。設計判断は公式Rev.Fを参照 |
| LTC4359 | `ADI_LTC4359_datasheet_RevE.pdf` | https://www.analog.com/media/en/technical-documentation/data-sheets/ltc4359.pdf | ローカルはRev.E。設計判断は公式Rev.Fを参照 |
| HT75xx-1 | `Holtek_HT75xx-1_datasheet.pdf` | https://www.holtek.com/webapi/116711/HT75xx-1v130.pdf | 公式PDF |
| MAU100 series | `MINMAX_MAU100_datasheet.pdf` | https://www.minmaxpower.com/storage/media/Product-MINMAX/MAU100/Document/MAU100_Datasheet.pdf | 公式PDF |
| SP3485 | `MaxLinear_SP3485_datasheet.pdf` | https://www.maxlinear.com/ds/sp3485.pdf | 公式PDF |
| SP481E/SP485E | `MaxLinear_SP481E_SP485E_datasheet.pdf` | https://www.maxlinear.com/ds/sp481e_sp485e.pdf | 公式PDF |
| IPT015N10NF2S | `Infineon_IPT015N10NF2S_datasheet.pdf` | https://www.infineon.com/part/IPT015N10NF2S | 公式PDF |
| 2N7002 | `onsemi_2N7002_datasheet.pdf` | https://www.onsemi.com/pdf/datasheet/2n7002-d.pdf | 公式PDF |
| Epson水晶型番体系 | `Epson_crystal_product_configuration.pdf` | https://www.epsondevice.com/crystal/en/support/product-no/pdf/pcs_xtal.pdf | Y1の正確な型番決定用 |
| Littelfuse SMDJ | 未保存 | https://www.littelfuse.com/assetdocs/littelfuse-tvs-diode-smdj-datasheet?assetguid=2d9772b3-822b-4362-914d-b74e0a706638 | 公式サーバーがCLI取得を403拒否するためリンクのみ |
| Jetson Orin Nano carrier | `NVIDIA_Jetson_Orin_Nano_carrier_spec.pdf` | https://developer.nvidia.com/embedded/downloads | 40-pin header照合用 |
| JST VH | `JST_VH_connector_datasheet.pdf` | https://www.jst-mfg.com/product/pdf/eng/eVH.pdf | 公式PDF |

## 注意

- `ADuM121N`、`SP485E`、`S9012`、水晶Y1は、回路図上の型番だけでは注文型番を一意に決められません。温度範囲、フェイルセーフ状態、パッケージ、負荷容量を含む完全なMPNが必要です。
- 抵抗、コンデンサ、インダクタにも定格電圧・許容損失・温度特性を含むMPNを割り当てないと、48V系と電源回路の最終判定はできません。
