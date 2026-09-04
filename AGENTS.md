# Pochi hardware workflow

## Execution host

- Treat this Mac workspace as the source of truth for code edits.
- Run all hardware-connected commands on the host reached with `ssh pochi`.
- The remote project directory is `/home/continue/Pochi`.
- Before building, flashing, testing hardware, or restarting servers, sync the local workspace to the remote directory with `rsync`. Do not use `--delete` unless the user explicitly requests deletion.
- Exclude generated and machine-local directories from synchronization: `.git/`, `.venv/`, `node_modules/`, `.pio/`, `dist/`, `.vinext/`, and `.next/`.

## Teensy and services

- The Teensy is attached to the remote host, normally as `/dev/ttyACM0`. Re-detect `/dev/ttyACM*` before flashing if the path is absent.
- Build and upload the normal firmware from `/home/continue/Pochi/pochi_hardware/firmware/teensy41_test` with `/home/continue/.platformio/penv/bin/pio run -t upload`.
- Keep torque off while deploying, flashing, reconnecting, or restarting services.
- The remote user services are `pochi-teensy-bridge.service`, `pochi-web-api.service`, and `pochi-web-ui.service`.
- After deployment, restart those services and verify `http://127.0.0.1:8765/health` on the remote host reports 12 motors and the IMU before enabling torque.
- Access the remote GUI through an SSH tunnel forwarding local ports 3000 and 8765 to the same remote ports. Do not expose the torque-control API directly to the LAN.

