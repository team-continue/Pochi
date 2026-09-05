# Pochi hardware workflow

## Execution host

- Treat `/home/continue/Pochi` on the Orin host reached with `ssh pochi` as the source of truth for code edits.
- Edit, build, flash, test hardware, and restart services directly on the Orin host.
- Do not synchronize the Mac workspace to the Orin host with `rsync` unless the user explicitly requests it.
- Keep generated and machine-local directories local to each host: `.git/`, `.venv/`, `node_modules/`, `.pio/`, `dist/`, `.vinext/`, and `.next/`.

## Teensy and services

- The Teensy is attached to the remote host, normally as `/dev/ttyACM0`. Re-detect `/dev/ttyACM*` before flashing if the path is absent.
- Build and upload the normal firmware from `/home/continue/Pochi/pochi_hardware/firmware/teensy41_test` with `/home/continue/.platformio/penv/bin/pio run -t upload`.
- Keep torque off while deploying, flashing, reconnecting, or restarting services.
- The remote user services are `pochi-teensy-bridge.service`, `pochi-web-api.service`, and `pochi-web-ui.service`.
- After deployment, restart those services and verify `http://127.0.0.1:8765/health` on the remote host reports 12 motors and the IMU before enabling torque.
- Access the remote GUI through an SSH tunnel forwarding local ports 3000 and 8765 to the same remote ports. Do not expose the torque-control API directly to the LAN.

