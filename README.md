# Pochi
CoREストライダー部門用のロボット「Pochi」のメインレポジトリです。

機体の構成は [Pochi の構成解説 — メカ・電装・制御](ROBOT_OVERVIEW.md) を参照してください。

詳細はドキュメントサイトを参照してください。

## 開発ルール
開発はブランチを切って Pull Request で main にマージしてください。main への直接コミットは禁止です。
ブランチ名: 機能追加は feature/hogehoge、バグ修正は fix/hogehoge
パッケージ名: pochi_hoge（タブ補完で一覧表示できるように）
Formatting (CI)
Push 時に CI が ament_clang_format を実行し、必要ならブランチへ自動コミットします。 main ブランチには自動 push しません。

## ビルド
``` shell
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src -i -y
colcon build --symlink-install
source install/setup.bash
```

## 起動方法
```
# シミュレータモード（デフォルト: /sim_odom使用）
ros2 launch core_launch navigation.launch.py

# シミュレータ + FAST-LIOオドメトリ
ros2 launch core_launch navigation.launch.py odom_source:=fastlio

# 実機モード（FAST-LIO + body_controller、TCP endpoint無し）
ros2 launch core_launch navigation.launch.py environment:=real

# 実機モード（GUI・ショートカットから）
# .bashrc が非対話シェルで早期 return するため、このラッパーを使用
ros2 run core_launch navigation.sh

# ボディコントローラ単体
ros2 launch core_launch navigation.launch.py environment:=real map_name:=curious_house.png

# 実機ハードウェア
ros2 launch core_hardware core_hardware.launch.py
```
Launch 引数の詳細はドキュメントサイトの「クイックスタート」を参照してください。


## 2026-10-14 の変更整理

作業版は `work/pochi-consolidation-20261014` に、CAD、起立・低速歩行、MPC・横ジャンプ、
LP-ACRLの過去実験、トルク余裕学習、歩行・停止の再学習を分けてコミットしています。
変更の記録は `pochi_rl/repository_state_20261014.json` を参照してください。

実際にSSH先で学習したソースとアセットは `archive/v19-ssh-20261014`、
Orinの実機コード・配布モデルは `archive/orin-hardware-20261014` に保存しています。
前者の作業ツリーは `.training-worktrees/v19-ssh-20261014/`、後者は
`.hardware-worktrees/orin-20261014/` です。SSH側の学習時のアセットと、この作業版の
CADアセットは異なるため、履歴を分けてそのまま残しています。

次の学習は利用許可が戻ったローカルGPUを使用します。大きい学習チェックポイント・
動画・CSVは各ホストの元の保存先に残し、実機起動に必要な小さいTorchScriptモデルと
契約は実機ブランチに含めています。機体固有のキャリブレーションはOrin上で保持します。
今回の整理は学習開始、実機の制御コード変更、サービス再起動を伴いません。
