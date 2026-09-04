'use client';

import {
  ContactShadows,
  Grid,
  Html,
  Line,
  OrbitControls,
} from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import {
  Activity,
  Box,
  Gauge,
  Power,
  Radio,
  RotateCcw,
  TriangleAlert,
} from 'lucide-react';
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Quaternion, Vector3 } from 'three';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';

type ConnectionState = 'connecting' | 'connected' | 'disconnected';

type MotorState = {
  id: number;
  name: string;
  label: string;
  leg: string;
  joint: string;
  positionRad: number | null;
  positionDeg: number | null;
  velocityRadS: number | null;
  torqueNm: number | null;
  targetRad: number;
  tempMosC: number | null;
  busVoltageV: number | null;
  ageMs: number | null;
  flags: number;
  faultCode: number;
  state: string;
};

type ImuState = {
  connected: boolean;
  rollDeg: number | null;
  pitchDeg: number | null;
  yawDeg: number | null;
  quaternionW: number | null;
  quaternionX: number | null;
  quaternionY: number | null;
  quaternionZ: number | null;
  ageMs: number | null;
  sampleCounter: number;
  accuracy: number;
};

type RobotState = {
  type: 'state';
  connected: boolean;
  connectedCount: number;
  expectedCount: number;
  canEnableTorque: boolean;
  unavailableMotorIds: number[];
  torqueEnabled: boolean;
  emergencyStop: boolean;
  stateAgeMs: number | null;
  motors: MotorState[];
  imu: ImuState;
  stats: {
    stateHz: number;
    rttMs: number;
    receivedPackets: number;
    droppedPackets: number;
    invalidPackets: number;
    lastError: string;
  };
};

const JOINTS = [
  ['front_left_calf', 'FL Calf', 'front_left', 'calf', 0],
  ['front_left_thigh', 'FL Thigh', 'front_left', 'thigh', 1],
  ['front_left_hip', 'FL Hip', 'front_left', 'hip', 2],
  ['rear_left_calf', 'RL Calf', 'rear_left', 'calf', 3],
  ['rear_left_thigh', 'RL Thigh', 'rear_left', 'thigh', 4],
  ['rear_left_hip', 'RL Hip', 'rear_left', 'hip', 5],
  ['rear_right_calf', 'RR Calf', 'rear_right', 'calf', 6],
  ['rear_right_thigh', 'RR Thigh', 'rear_right', 'thigh', 7],
  ['rear_right_hip', 'RR Hip', 'rear_right', 'hip', 8],
  ['front_right_calf', 'FR Calf', 'front_right', 'calf', 9],
  ['front_right_thigh', 'FR Thigh', 'front_right', 'thigh', 10],
  ['front_right_hip', 'FR Hip', 'front_right', 'hip', 11],
] as const;

const EMPTY_MOTORS: MotorState[] = JOINTS.map(
  ([name, label, leg, joint, id]) => ({
    id,
    name,
    label,
    leg,
    joint,
    positionRad: null,
    positionDeg: null,
    velocityRadS: null,
    torqueNm: null,
    targetRad: 0,
    tempMosC: null,
    busVoltageV: null,
    ageMs: null,
    flags: 0,
    faultCode: 0,
    state: 'NO DATA',
  }),
);

const EMPTY_IMU: ImuState = {
  connected: false,
  rollDeg: null,
  pitchDeg: null,
  yawDeg: null,
  quaternionW: null,
  quaternionX: null,
  quaternionY: null,
  quaternionZ: null,
  ageMs: null,
  sampleCounter: 0,
  accuracy: 0,
};

const EMPTY_STATE: RobotState = {
  type: 'state',
  connected: false,
  connectedCount: 0,
  expectedCount: 12,
  canEnableTorque: false,
  unavailableMotorIds: Array.from({ length: 12 }, (_, index) => index),
  torqueEnabled: false,
  emergencyStop: false,
  stateAgeMs: null,
  motors: EMPTY_MOTORS,
  imu: EMPTY_IMU,
  stats: {
    stateHz: 0,
    rttMs: 0,
    receivedPackets: 0,
    droppedPackets: 0,
    invalidPackets: 0,
    lastError: '',
  },
};

const LIMITS_DEG: Record<string, [number, number]> = {
  hip: [-30, 90],
  thigh: [-90, 90],
  calf: [-135, 135],
};

// Motor mounting direction affects only the robot drawing. Encoder numbers and
// targets stay in the Teensy's joint coordinate system.
const VIEWER_REVERSED_MOTOR_IDS = new Set([0, 1, 3, 4, 5, 6, 7, 9, 10, 11]);

function viewerAngle(motor: MotorState): number {
  const angle = motor.positionRad ?? 0;
  return VIEWER_REVERSED_MOTOR_IDS.has(motor.id) ? -angle : angle;
}

function degrees(rad: number | null): number | null {
  return rad === null ? null : (rad * 180) / Math.PI;
}

function radians(deg: number): number {
  return (deg * Math.PI) / 180;
}

function numberText(value: number | null, digits = 1): string {
  return value === null || !Number.isFinite(value)
    ? '—'
    : value.toFixed(digits);
}

function torqueText(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '—';
  const rounded = Math.abs(value) < 0.005 ? 0 : value;
  return `${rounded > 0 ? '+' : ''}${rounded.toFixed(2)}`;
}

function faultCodeText(code: number): string {
  return `0x${code.toString(16).toUpperCase().padStart(2, '0')}`;
}

function useRobotSocket() {
  const [state, setState] = useState<RobotState>(EMPTY_STATE);
  const [connection, setConnection] = useState<ConnectionState>('connecting');
  const [controlError, setControlError] = useState('');
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let active = true;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      if (!active) return;
      setConnection('connecting');
      const configured = process.env.NEXT_PUBLIC_POCHI_WS_URL;
      const url =
        configured ?? `ws://${window.location.hostname || '127.0.0.1'}:8765/ws`;
      const socket = new WebSocket(url);
      socketRef.current = socket;
      socket.onopen = () => active && setConnection('connected');
      socket.onmessage = (event) => {
        try {
          const next = JSON.parse(event.data) as
            | RobotState
            | { type: 'error'; message: string };
          if (next.type === 'state') {
            setState(next);
          } else {
            setControlError(next.message);
          }
        } catch {
          // Keep the last valid telemetry snapshot.
        }
      };
      socket.onerror = () => socket.close();
      socket.onclose = () => {
        if (!active) return;
        setConnection('disconnected');
        setState((previous) => ({
          ...previous,
          connected: false,
          torqueEnabled: false,
        }));
        retryTimer = setTimeout(connect, 1200);
      };
    };

    connect();
    return () => {
      active = false;
      if (retryTimer) clearTimeout(retryTimer);
      socketRef.current?.close();
    };
  }, []);

  const send = useCallback((message: object) => {
    setControlError('');
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message));
    }
  }, []);

  return { state, connection, controlError, send };
}

function Bone({ length, radius = 0.075 }: { length: number; radius?: number }) {
  return (
    <mesh position={[0, -length / 2, 0]} castShadow>
      <cylinderGeometry args={[radius, radius * 0.88, length, 16]} />
      <meshStandardMaterial color="#cbd5e1" metalness={0.68} roughness={0.28} />
    </mesh>
  );
}

type TorqueAxis = 'x' | 'z';
type Point3 = [number, number, number];

function torqueArcPoint(
  axis: TorqueAxis,
  angle: number,
  radius: number,
): Point3 {
  if (axis === 'x')
    return [0, Math.cos(angle) * radius, Math.sin(angle) * radius];
  return [Math.cos(angle) * radius, Math.sin(angle) * radius, 0];
}

function TorqueIndicator({
  motor,
  axis,
  directionScale,
  labelPosition,
}: {
  motor: MotorState;
  axis: TorqueAxis;
  directionScale: number;
  labelPosition: Point3;
}) {
  const torque = motor.torqueNm;
  const viewerDirection = torque === null ? 0 : torque * directionScale;
  const direction =
    Math.abs(viewerDirection) < 0.02 ? 0 : Math.sign(viewerDirection);
  const color =
    direction > 0 ? '#38bdf8' : direction < 0 ? '#fb7185' : '#64748b';
  const radius = 0.19;
  const startAngle = -Math.PI * 0.68;
  const sweep = (direction || 1) * Math.PI * 1.36;
  const arc = useMemo(
    () =>
      Array.from({ length: 33 }, (_, index) =>
        torqueArcPoint(axis, startAngle + sweep * (index / 32), radius),
      ),
    [axis, startAngle, sweep],
  );
  const endAngle = startAngle + sweep;
  const arrowPosition = torqueArcPoint(axis, endAngle, radius);
  const tangent =
    axis === 'x'
      ? new Vector3(
          0,
          -Math.sin(endAngle) * (direction || 1),
          Math.cos(endAngle) * (direction || 1),
        )
      : new Vector3(
          -Math.sin(endAngle) * (direction || 1),
          Math.cos(endAngle) * (direction || 1),
          0,
        );
  const arrowQuaternion = new Quaternion().setFromUnitVectors(
    new Vector3(0, 1, 0),
    tangent.normalize(),
  );
  const magnitude = torque === null ? 0 : Math.min(Math.abs(torque) / 20, 1);
  const labelTone = motor.faultCode
    ? 'fault'
    : direction > 0
      ? 'positive'
      : direction < 0
        ? 'negative'
        : 'idle';

  return (
    <group renderOrder={20}>
      <Line
        points={arc}
        color={color}
        lineWidth={1.5 + magnitude * 2.5}
        opacity={torque === null ? 0.28 : 0.72 + magnitude * 0.28}
        transparent
        depthTest={false}
      />
      {direction !== 0 && (
        <mesh
          position={arrowPosition}
          quaternion={arrowQuaternion}
          renderOrder={21}
        >
          <coneGeometry args={[0.045 + magnitude * 0.018, 0.13, 14]} />
          <meshBasicMaterial
            color={color}
            depthTest={false}
            transparent
            opacity={0.95}
          />
        </mesh>
      )}
      <Html
        position={labelPosition}
        center
        zIndexRange={[10, 0]}
        style={{ pointerEvents: 'none' }}
      >
        <div className={`joint-torque-label ${labelTone}`}>
          <span>ID {motor.id}</span>
          <strong>{torqueText(torque)}</strong>
          <small>Nm</small>
        </div>
      </Html>
    </group>
  );
}

function JointNode({
  motor,
  selected,
  axis,
  directionScale,
  labelPosition,
  onSelect,
}: {
  motor: MotorState;
  selected: boolean;
  axis: TorqueAxis;
  directionScale: number;
  labelPosition: Point3;
  onSelect: (id: number) => void;
}) {
  const online =
    motor.positionRad !== null &&
    (motor.state === 'ENCODER LIVE' || motor.state === 'MIT ENABLED');
  const color = motor.faultCode
    ? '#fb7185'
    : selected
      ? '#38bdf8'
      : online
        ? '#34d399'
        : '#475569';
  return (
    <group>
      <mesh
        castShadow
        onClick={(event) => {
          event.stopPropagation();
          onSelect(motor.id);
        }}
      >
        <sphereGeometry args={[selected ? 0.135 : 0.105, 20, 20]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={selected ? 0.65 : 0.12}
        />
      </mesh>
      <TorqueIndicator
        motor={motor}
        axis={axis}
        directionScale={directionScale}
        labelPosition={labelPosition}
      />
    </group>
  );
}

function RobotLeg({
  motors,
  anchor,
  side,
  selectedId,
  onSelect,
}: {
  motors: MotorState[];
  anchor: [number, number, number];
  side: 1 | -1;
  selectedId: number;
  onSelect: (id: number) => void;
}) {
  const hip = motors.find((motor) => motor.joint === 'hip');
  const thigh = motors.find((motor) => motor.joint === 'thigh');
  const calf = motors.find((motor) => motor.joint === 'calf');
  if (!hip || !thigh || !calf) return null;
  const hipAngle = viewerAngle(hip);
  const thighAngle = viewerAngle(thigh);
  const calfAngle = viewerAngle(calf);
  const labelX = Math.sign(anchor[0]) * 0.28;
  const hipLabelPosition: Point3 = [labelX, 0.24, side * 0.16];
  const thighLabelPosition: Point3 = [labelX, 0.07, side * 0.09];
  const calfLabelPosition: Point3 = [labelX, -0.08, side * 0.06];
  const torqueDirection = (motor: MotorState) =>
    side * (VIEWER_REVERSED_MOTOR_IDS.has(motor.id) ? -1 : 1);

  return (
    <group position={anchor}>
      <group rotation={[side * hipAngle, 0, 0]}>
        <JointNode
          motor={hip}
          selected={selectedId === hip.id}
          axis="x"
          directionScale={torqueDirection(hip)}
          labelPosition={hipLabelPosition}
          onSelect={onSelect}
        />
        <mesh
          position={[0, -0.04, side * 0.2]}
          rotation={[Math.PI / 2, 0, 0]}
          castShadow
        >
          <cylinderGeometry args={[0.07, 0.07, 0.4, 16]} />
          <meshStandardMaterial
            color="#94a3b8"
            metalness={0.65}
            roughness={0.3}
          />
        </mesh>
        <group
          position={[0, 0, side * 0.4]}
          rotation={[0, 0, side * thighAngle]}
        >
          <JointNode
            motor={thigh}
            selected={selectedId === thigh.id}
            axis="z"
            directionScale={torqueDirection(thigh)}
            labelPosition={thighLabelPosition}
            onSelect={onSelect}
          />
          <Bone length={0.88} radius={0.085} />
          <group position={[0, -0.88, 0]} rotation={[0, 0, side * calfAngle]}>
            <JointNode
              motor={calf}
              selected={selectedId === calf.id}
              axis="z"
              directionScale={torqueDirection(calf)}
              labelPosition={calfLabelPosition}
              onSelect={onSelect}
            />
            <Bone length={0.92} radius={0.068} />
            <mesh position={[0.09, -0.92, 0]} castShadow>
              <boxGeometry args={[0.34, 0.08, 0.16]} />
              <meshStandardMaterial color="#334155" roughness={0.72} />
            </mesh>
          </group>
        </group>
      </group>
    </group>
  );
}

function RobotModel({
  motors,
  imu,
  selectedId,
  onSelect,
}: {
  motors: MotorState[];
  imu: ImuState;
  selectedId: number;
  onSelect: (id: number) => void;
}) {
  const byLeg = useMemo(
    () =>
      motors.reduce<Record<string, MotorState[]>>((groups, motor) => {
        (groups[motor.leg] ??= []).push(motor);
        return groups;
      }, {}),
    [motors],
  );
  const quaternion: [number, number, number, number] =
    imu.connected &&
    [imu.quaternionX, imu.quaternionY, imu.quaternionZ, imu.quaternionW].every(
      (value) => value !== null,
    )
      ? [imu.quaternionX!, imu.quaternionY!, imu.quaternionZ!, imu.quaternionW!]
      : [0, 0, 0, 1];

  return (
    <group position={[0, 0.25, 0]} quaternion={quaternion}>
      <mesh position={[0, 1.55, 0]} castShadow receiveShadow>
        <boxGeometry args={[3.1, 0.25, 1.45]} />
        <meshStandardMaterial
          color="#172033"
          metalness={0.55}
          roughness={0.38}
        />
      </mesh>
      <mesh position={[0.55, 1.69, 0]} castShadow>
        <boxGeometry args={[1.3, 0.12, 0.78]} />
        <meshStandardMaterial
          color="#24324b"
          metalness={0.42}
          roughness={0.4}
        />
      </mesh>
      <RobotLeg
        motors={byLeg.front_left ?? []}
        anchor={[1.15, 1.48, -0.68]}
        side={-1}
        selectedId={selectedId}
        onSelect={onSelect}
      />
      <RobotLeg
        motors={byLeg.front_right ?? []}
        anchor={[1.15, 1.48, 0.68]}
        side={1}
        selectedId={selectedId}
        onSelect={onSelect}
      />
      <RobotLeg
        motors={byLeg.rear_left ?? []}
        anchor={[-1.15, 1.48, -0.68]}
        side={-1}
        selectedId={selectedId}
        onSelect={onSelect}
      />
      <RobotLeg
        motors={byLeg.rear_right ?? []}
        anchor={[-1.15, 1.48, 0.68]}
        side={1}
        selectedId={selectedId}
        onSelect={onSelect}
      />
    </group>
  );
}

function RobotViewport({
  motors,
  imu,
  selectedId,
  onSelect,
}: {
  motors: MotorState[];
  imu: ImuState;
  selectedId: number;
  onSelect: (id: number) => void;
}) {
  return (
    <Canvas
      shadows="basic"
      camera={{ position: [5.8, 4.1, 5.8], fov: 38, near: 0.1, far: 100 }}
      dpr={[1, 1.7]}
    >
      <color attach="background" args={['#080c14']} />
      <fog attach="fog" args={['#080c14', 9, 19]} />
      <ambientLight intensity={0.78} />
      <directionalLight
        position={[4, 8, 4]}
        intensity={2.2}
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <pointLight
        position={[-4, 3, -3]}
        intensity={5}
        color="#38bdf8"
        distance={9}
      />
      <Suspense fallback={null}>
        <RobotModel
          motors={motors}
          imu={imu}
          selectedId={selectedId}
          onSelect={onSelect}
        />
        <ContactShadows
          position={[0, 0.02, 0]}
          opacity={0.48}
          scale={9}
          blur={2.8}
          far={5}
        />
        <Grid
          position={[0, 0, 0]}
          args={[12, 12]}
          cellSize={0.4}
          cellThickness={0.55}
          cellColor="#1e293b"
          sectionSize={2}
          sectionThickness={0.9}
          sectionColor="#334155"
          fadeDistance={10}
          fadeStrength={1.2}
          infiniteGrid
        />
      </Suspense>
      <OrbitControls
        makeDefault
        target={[0, 1.0, 0]}
        minDistance={4.5}
        maxDistance={11}
        maxPolarAngle={Math.PI / 2.05}
      />
    </Canvas>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function Home() {
  const { state, connection, controlError, send } = useRobotSocket();
  const [selectedId, setSelectedId] = useState(0);
  const selected =
    state.motors.find((motor) => motor.id === selectedId) ?? state.motors[0];
  const [localTargetDeg, setLocalTargetDeg] = useState(0);
  const lastRemoteTarget = useRef<number | null>(null);
  const hardwareLive = connection === 'connected' && state.connected;
  const allMotorsLive =
    hardwareLive && state.connectedCount === state.expectedCount;
  const faultedMotors = state.motors.filter((motor) => motor.faultCode !== 0);
  const faultSummary = !hardwareLive
    ? 'NO DATA'
    : faultedMotors.length
      ? faultedMotors
          .map((motor) => `ID ${motor.id} ${faultCodeText(motor.faultCode)}`)
          .join(' · ')
      : 'NONE';
  const limits = LIMITS_DEG[selected.joint] ?? [-180, 180];

  useEffect(() => {
    const remote = degrees(selected.targetRad) ?? 0;
    if (lastRemoteTarget.current !== remote) {
      lastRemoteTarget.current = remote;
      setLocalTargetDeg(remote);
    }
  }, [selected.id, selected.targetRad]);

  const setTarget = (value: number) => {
    setLocalTargetDeg(value);
    send({ type: 'target', motorId: selected.id, positionRad: radians(value) });
  };

  const toggleTorque = () =>
    send({ type: 'torque', enabled: !state.torqueEnabled });

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">P</div>
          <div>
            <h1>Pochi Control</h1>
            <p>Encoder + IMU + MIT control</p>
          </div>
        </div>
        <div className="status-cluster">
          <Badge
            variant="outline"
            className={hardwareLive ? 'status-ok' : 'status-warn'}
          >
            <Radio /> {hardwareLive ? 'Hardware live' : 'Waiting for hardware'}
          </Badge>
          <Badge
            variant="outline"
            className={
              state.torqueEnabled ? 'status-torque-on' : 'status-torque-off'
            }
          >
            <Power /> Torque {state.torqueEnabled ? 'ON' : 'OFF'}
          </Badge>
          <span className="top-metric">
            <Activity /> {state.connectedCount}/{state.expectedCount} motors
          </span>
          <span className="top-metric">
            {state.stats.stateHz.toFixed(0)} Hz
          </span>
          <span className="top-metric">
            <Gauge /> {numberText(state.stats.rttMs, 2)} ms
          </span>
          <Button
            size="lg"
            variant={state.torqueEnabled ? 'destructive' : 'default'}
            className={state.torqueEnabled ? 'torque-on' : 'torque-off'}
            disabled={
              connection !== 'connected' ||
              (!state.canEnableTorque && !state.torqueEnabled)
            }
            onClick={toggleTorque}
          >
            <Power /> {state.torqueEnabled ? 'TURN OFF' : 'TURN ON'}
          </Button>
        </div>
      </header>

      <section className="workspace">
        <div className="viewer-panel">
          <div className="viewer-title">
            <div>
              <span className="eyebrow">LIVE POSE</span>
              <h2>Encoder + body attitude</h2>
            </div>
            <p>Drag to orbit · scroll to zoom · click a joint to inspect</p>
          </div>
          <div className="viewport">
            <RobotViewport
              motors={state.motors}
              imu={state.imu}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
            <div className="motor-status-overlay">
              <div>
                <span>Motors</span>
                <strong className={allMotorsLive ? 'ok' : 'warn'}>
                  {hardwareLive
                    ? `${state.connectedCount} / ${state.expectedCount}`
                    : 'NO DATA'}
                </strong>
              </div>
              <div>
                <span>Torque</span>
                <strong className={state.torqueEnabled ? 'danger' : 'safe'}>
                  {state.torqueEnabled ? 'ON' : 'OFF'}
                </strong>
              </div>
              <div>
                <span>Faults</span>
                <strong
                  className={
                    faultedMotors.length
                      ? 'danger'
                      : hardwareLive
                        ? 'ok'
                        : 'warn'
                  }
                >
                  {faultSummary}
                </strong>
              </div>
            </div>
            <div
              className={
                state.imu.connected ? 'imu-overlay online' : 'imu-overlay'
              }
            >
              <span>IMU orientation</span>
              <strong>R {numberText(state.imu.rollDeg)}°</strong>
              <strong>P {numberText(state.imu.pitchDeg)}°</strong>
              <strong>Y {numberText(state.imu.yawDeg)}°</strong>
            </div>
            {!hardwareLive && (
              <div className="offline-overlay">
                <TriangleAlert /> Live telemetry unavailable — neutral pose
                shown
              </div>
            )}
          </div>
        </div>

        <aside className="telemetry-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">ENCODER DETAIL</span>
              <h2>{selected.label}</h2>
            </div>
            <span className="motor-id">
              ID {selected.id.toString().padStart(2, '0')}
            </span>
          </div>

          <div className="encoder-card">
            <span>Mechanical position</span>
            <strong>{numberText(selected.positionDeg, 2)}°</strong>
            <small>{numberText(selected.positionRad, 4)} rad</small>
          </div>

          <div className="position-card">
            <div className="position-readout">
              <div>
                <span>Target angle</span>
                <strong>{localTargetDeg.toFixed(1)}°</strong>
              </div>
              <Button
                variant="ghost"
                size="icon"
                title="Set target to current angle"
                onClick={() => setTarget(selected.positionDeg ?? 0)}
              >
                <RotateCcw />
              </Button>
            </div>
            <Slider
              value={[localTargetDeg]}
              min={limits[0]}
              max={limits[1]}
              step={0.5}
              disabled={!state.torqueEnabled || !state.connected}
              onValueChange={(value) =>
                setTarget(typeof value === 'number' ? value : value[0])
              }
              aria-label={`${selected.label} target angle`}
            />
            <div className="slider-limits">
              <span>{limits[0]}°</span>
              <span>Actual {numberText(selected.positionDeg)}°</span>
              <span>{limits[1]}°</span>
            </div>
            {!state.torqueEnabled && (
              <p className="safety-note">
                Torque starts OFF. Turning it ON initializes every target from
                the live pose.
              </p>
            )}
            {!state.canEnableTorque && state.connected && (
              <p className="safety-note">
                Unavailable or outside limits: ID{' '}
                {state.unavailableMotorIds.join(', ')}
              </p>
            )}
            {controlError && <p className="control-error">{controlError}</p>}
          </div>

          <div className="joint-grid" aria-label="Encoder selection">
            {state.motors.map((motor) => (
              <button
                key={motor.id}
                className={
                  motor.id === selectedId
                    ? 'joint-button selected'
                    : 'joint-button'
                }
                onClick={() => setSelectedId(motor.id)}
                title={`${motor.label}, CAN ID ${motor.id}`}
              >
                <span>
                  <b>{motor.id}</b>
                  {motor.label}
                </span>
                <strong>{numberText(motor.positionDeg, 1)}°</strong>
                <i
                  className={
                    motor.faultCode
                      ? 'fault'
                      : motor.state === 'ENCODER LIVE' ||
                          motor.state === 'MIT ENABLED'
                        ? 'online'
                        : ''
                  }
                />
              </button>
            ))}
          </div>

          <div className="imu-card">
            <div className="section-title">
              <Box />
              <span>IMU attitude</span>
              <i className={state.imu.connected ? 'online' : ''} />
            </div>
            <div className="metrics-grid three">
              <Metric
                label="Roll"
                value={`${numberText(state.imu.rollDeg, 2)}°`}
              />
              <Metric
                label="Pitch"
                value={`${numberText(state.imu.pitchDeg, 2)}°`}
              />
              <Metric
                label="Yaw"
                value={`${numberText(state.imu.yawDeg, 2)}°`}
              />
            </div>
            <div className="quaternion-row">
              <span>Quaternion</span>
              <code>
                {numberText(state.imu.quaternionW, 4)},{' '}
                {numberText(state.imu.quaternionX, 4)},{' '}
                {numberText(state.imu.quaternionY, 4)},{' '}
                {numberText(state.imu.quaternionZ, 4)}
              </code>
            </div>
          </div>

          <div className="metrics-grid">
            <Metric
              label="Position"
              value={`${numberText(selected.positionDeg, 2)}°`}
            />
            <Metric
              label="Velocity"
              value={`${numberText(selected.velocityRadS, 2)} rad/s`}
            />
            <Metric
              label="Torque"
              value={`${numberText(selected.torqueNm, 2)} Nm`}
            />
            <Metric
              label="Fault"
              value={
                selected.faultCode
                  ? faultCodeText(selected.faultCode)
                  : hardwareLive
                    ? 'None'
                    : 'No data'
              }
            />
            <Metric
              label="MOS temp"
              value={`${numberText(selected.tempMosC)} °C`}
            />
            <Metric
              label="Encoder age"
              value={`${numberText(selected.ageMs, 2)} ms`}
            />
            <Metric
              label="IMU age"
              value={`${numberText(state.imu.ageMs, 2)} ms`}
            />
          </div>

          <div className="state-footer">
            <div>
              <span
                className={
                  selected.faultCode
                    ? 'state-dot fault'
                    : selected.state === 'ENCODER LIVE' ||
                        selected.state === 'MIT ENABLED'
                      ? 'state-dot online'
                      : 'state-dot'
                }
              />
              {selected.state}
            </div>
            <span>
              RX {state.stats.receivedPackets.toLocaleString()} · Lost{' '}
              {state.stats.droppedPackets.toLocaleString()}
            </span>
          </div>
        </aside>
      </section>
    </main>
  );
}
