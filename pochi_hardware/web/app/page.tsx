'use client';

import { Canvas } from '@react-three/fiber';
import { ContactShadows, Grid, OrbitControls } from '@react-three/drei';
import { Activity, Gauge, Power, Radio, RotateCcw, TriangleAlert } from 'lucide-react';
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';

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

type RobotState = {
  type: 'state';
  connected: boolean;
  canEnableTorque: boolean;
  unavailableMotorIds: number[];
  torqueEnabled: boolean;
  emergencyStop: boolean;
  stateAgeMs: number | null;
  motors: MotorState[];
  imu: {
    connected: boolean;
    rollDeg: number | null;
    pitchDeg: number | null;
    yawDeg: number | null;
  };
  stats: {
    stateHz: number;
    rttMs: number;
    receivedPackets: number;
    droppedPackets: number;
  };
};

const JOINTS = [
  ['front_left_hip', 'FL Hip', 'front_left', 'hip', 1],
  ['front_left_thigh', 'FL Thigh', 'front_left', 'thigh', 2],
  ['front_left_calf', 'FL Calf', 'front_left', 'calf', 3],
  ['front_right_hip', 'FR Hip', 'front_right', 'hip', 4],
  ['front_right_thigh', 'FR Thigh', 'front_right', 'thigh', 5],
  ['front_right_calf', 'FR Calf', 'front_right', 'calf', 6],
  ['rear_left_hip', 'RL Hip', 'rear_left', 'hip', 7],
  ['rear_left_thigh', 'RL Thigh', 'rear_left', 'thigh', 8],
  ['rear_left_calf', 'RL Calf', 'rear_left', 'calf', 9],
  ['rear_right_hip', 'RR Hip', 'rear_right', 'hip', 10],
  ['rear_right_thigh', 'RR Thigh', 'rear_right', 'thigh', 11],
  ['rear_right_calf', 'RR Calf', 'rear_right', 'calf', 12],
] as const;

const EMPTY_MOTORS: MotorState[] = JOINTS.map(([name, label, leg, joint, id]) => ({
  id,
  name,
  label,
  leg,
  joint,
  positionRad: 0,
  velocityRadS: null,
  torqueNm: null,
  targetRad: 0,
  tempMosC: null,
  busVoltageV: null,
  ageMs: null,
  flags: 0,
  faultCode: 0,
  state: 'NO DATA',
}));

const EMPTY_STATE: RobotState = {
  type: 'state',
  connected: false,
  canEnableTorque: false,
  unavailableMotorIds: Array.from({ length: 12 }, (_, index) => index + 1),
  torqueEnabled: false,
  emergencyStop: false,
  stateAgeMs: null,
  motors: EMPTY_MOTORS,
  imu: { connected: false, rollDeg: null, pitchDeg: null, yawDeg: null },
  stats: { stateHz: 0, rttMs: 0, receivedPackets: 0, droppedPackets: 0 },
};

const LIMITS_DEG: Record<string, [number, number]> = {
  hip: [-55, 55],
  thigh: [-150, 150],
  calf: [-165, 25],
};

function degrees(rad: number | null): number | null {
  return rad === null ? null : (rad * 180) / Math.PI;
}

function radians(deg: number): number {
  return (deg * Math.PI) / 180;
}

function numberText(value: number | null, digits = 1): string {
  return value === null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
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
      const url = configured ?? `ws://${window.location.hostname || '127.0.0.1'}:8765/ws`;
      const socket = new WebSocket(url);
      socketRef.current = socket;
      socket.onopen = () => active && setConnection('connected');
      socket.onmessage = (event) => {
        try {
          const next = JSON.parse(event.data) as RobotState | { type: 'error'; message: string };
          if (next.type === 'state') {
            setState(next);
          } else if (next.type === 'error') {
            setControlError(next.message);
          }
        } catch {
          // Ignore malformed diagnostic messages without interrupting live control.
        }
      };
      socket.onerror = () => socket.close();
      socket.onclose = () => {
        if (!active) return;
        setConnection('disconnected');
        setState((previous) => ({ ...previous, connected: false, torqueEnabled: false }));
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

function JointNode({
  motor,
  selected,
  onSelect,
}: {
  motor: MotorState;
  selected: boolean;
  onSelect: (id: number) => void;
}) {
  const color = motor.faultCode
    ? '#fb7185'
    : selected
      ? '#38bdf8'
      : motor.state.includes('CONNECTED') || motor.state.includes('ENABLED')
        ? '#34d399'
        : '#64748b';
  return (
    <mesh
      castShadow
      onClick={(event) => {
        event.stopPropagation();
        onSelect(motor.id);
      }}
    >
      <sphereGeometry args={[selected ? 0.135 : 0.105, 20, 20]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={selected ? 0.65 : 0.12} />
    </mesh>
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
  const [hip, thigh, calf] = motors;
  const hipAngle = hip.positionRad ?? 0;
  const thighAngle = thigh.positionRad ?? 0;
  const calfAngle = calf.positionRad ?? 0;

  return (
    <group position={anchor}>
      <group rotation={[side * hipAngle, 0, 0]}>
        <JointNode motor={hip} selected={selectedId === hip.id} onSelect={onSelect} />
        <mesh position={[0, -0.04, side * 0.2]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.07, 0.07, 0.4, 16]} />
          <meshStandardMaterial color="#94a3b8" metalness={0.65} roughness={0.3} />
        </mesh>
        <group position={[0, 0, side * 0.4]} rotation={[0, 0, side * thighAngle]}>
          <JointNode motor={thigh} selected={selectedId === thigh.id} onSelect={onSelect} />
          <Bone length={0.88} radius={0.085} />
          <group position={[0, -0.88, 0]} rotation={[0, 0, side * calfAngle]}>
            <JointNode motor={calf} selected={selectedId === calf.id} onSelect={onSelect} />
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
  selectedId,
  onSelect,
}: {
  motors: MotorState[];
  selectedId: number;
  onSelect: (id: number) => void;
}) {
  const byLeg = useMemo(() => {
    return motors.reduce<Record<string, MotorState[]>>((groups, motor) => {
      (groups[motor.leg] ??= []).push(motor);
      return groups;
    }, {});
  }, [motors]);

  return (
    <group position={[0, 0.25, 0]}>
      <mesh position={[0, 1.55, 0]} castShadow receiveShadow>
        <boxGeometry args={[3.1, 0.25, 1.45]} />
        <meshStandardMaterial color="#172033" metalness={0.55} roughness={0.38} />
      </mesh>
      <mesh position={[0.55, 1.69, 0]} castShadow>
        <boxGeometry args={[1.3, 0.12, 0.78]} />
        <meshStandardMaterial color="#24324b" metalness={0.42} roughness={0.4} />
      </mesh>
      <RobotLeg motors={byLeg.front_left} anchor={[1.15, 1.48, 0.68]} side={1} selectedId={selectedId} onSelect={onSelect} />
      <RobotLeg motors={byLeg.front_right} anchor={[1.15, 1.48, -0.68]} side={-1} selectedId={selectedId} onSelect={onSelect} />
      <RobotLeg motors={byLeg.rear_left} anchor={[-1.15, 1.48, 0.68]} side={1} selectedId={selectedId} onSelect={onSelect} />
      <RobotLeg motors={byLeg.rear_right} anchor={[-1.15, 1.48, -0.68]} side={-1} selectedId={selectedId} onSelect={onSelect} />
    </group>
  );
}

function RobotViewport({ motors, selectedId, onSelect }: { motors: MotorState[]; selectedId: number; onSelect: (id: number) => void }) {
  return (
    <Canvas shadows="basic" camera={{ position: [5.8, 4.1, 5.8], fov: 38, near: 0.1, far: 100 }} dpr={[1, 1.7]}>
      <color attach="background" args={['#080c14']} />
      <fog attach="fog" args={['#080c14', 9, 19]} />
      <ambientLight intensity={0.78} />
      <directionalLight position={[4, 8, 4]} intensity={2.2} castShadow shadow-mapSize={[1024, 1024]} />
      <pointLight position={[-4, 3, -3]} intensity={5} color="#38bdf8" distance={9} />
      <Suspense fallback={null}>
        <RobotModel motors={motors} selectedId={selectedId} onSelect={onSelect} />
        <ContactShadows position={[0, 0.02, 0]} opacity={0.48} scale={9} blur={2.8} far={5} />
        <Grid position={[0, 0, 0]} args={[12, 12]} cellSize={0.4} cellThickness={0.55} cellColor="#1e293b" sectionSize={2} sectionThickness={0.9} sectionColor="#334155" fadeDistance={10} fadeStrength={1.2} infiniteGrid />
      </Suspense>
      <OrbitControls makeDefault target={[0, 1.0, 0]} minDistance={4.5} maxDistance={11} maxPolarAngle={Math.PI / 2.05} />
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
  const [selectedId, setSelectedId] = useState(1);
  const selected = state.motors.find((motor) => motor.id === selectedId) ?? state.motors[0];
  const [localTargetDeg, setLocalTargetDeg] = useState(0);
  const lastRemoteTarget = useRef<number | null>(null);
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

  const toggleTorque = () => send({ type: 'torque', enabled: !state.torqueEnabled });

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">P</div>
          <div>
            <h1>Pochi Control</h1>
            <p>12-axis live hardware viewer</p>
          </div>
        </div>
        <div className="status-cluster">
          <Badge variant="outline" className={connection === 'connected' && state.connected ? 'status-ok' : 'status-warn'}>
            <Radio /> {connection === 'connected' && state.connected ? 'Hardware live' : 'Waiting for hardware'}
          </Badge>
          <span className="top-metric"><Activity /> {state.stats.stateHz.toFixed(0)} Hz</span>
          <span className="top-metric"><Gauge /> {numberText(state.stats.rttMs, 2)} ms</span>
          <Button
            size="lg"
            variant={state.torqueEnabled ? 'destructive' : 'default'}
            className={state.torqueEnabled ? 'torque-on' : 'torque-off'}
            disabled={connection !== 'connected' || (!state.canEnableTorque && !state.torqueEnabled)}
            onClick={toggleTorque}
          >
            <Power /> {state.torqueEnabled ? 'TORQUE OFF' : 'TORQUE ON'}
          </Button>
        </div>
      </header>

      <section className="workspace">
        <div className="viewer-panel">
          <div className="viewer-title">
            <div>
              <span className="eyebrow">LIVE POSE</span>
              <h2>Robot state</h2>
            </div>
            <p>Drag to orbit · scroll to zoom · click a joint to select</p>
          </div>
          <div className="viewport">
            <RobotViewport motors={state.motors} selectedId={selectedId} onSelect={setSelectedId} />
            <div className="imu-overlay">
              <span>IMU orientation</span>
              <strong>R {numberText(state.imu.rollDeg)}°</strong>
              <strong>P {numberText(state.imu.pitchDeg)}°</strong>
              <strong>Y {numberText(state.imu.yawDeg)}°</strong>
            </div>
            {!state.connected && (
              <div className="offline-overlay"><TriangleAlert /> Live state unavailable — neutral pose shown</div>
            )}
          </div>
        </div>

        <aside className="control-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">JOINT CONTROL</span>
              <h2>{selected.label}</h2>
            </div>
            <span className="motor-id">ID {selected.id.toString().padStart(2, '0')}</span>
          </div>

          <div className="joint-grid" aria-label="Joint selection">
            {state.motors.map((motor) => (
              <button
                key={motor.id}
                className={motor.id === selectedId ? 'joint-button selected' : 'joint-button'}
                onClick={() => setSelectedId(motor.id)}
                title={`${motor.label}, CAN ID ${motor.id}`}
              >
                <span>{motor.label}</span>
                <i className={motor.faultCode ? 'fault' : motor.state.includes('CONNECTED') || motor.state.includes('ENABLED') ? 'online' : ''} />
              </button>
            ))}
          </div>

          <div className="position-card">
            <div className="position-readout">
              <div>
                <span>Target angle</span>
                <strong>{localTargetDeg.toFixed(1)}°</strong>
              </div>
              <Button variant="ghost" size="icon" title="Set target to current angle" onClick={() => setTarget(degrees(selected.positionRad) ?? 0)}>
                <RotateCcw />
              </Button>
            </div>
            <Slider
              value={[localTargetDeg]}
              min={limits[0]}
              max={limits[1]}
              step={0.5}
              disabled={!state.torqueEnabled || !state.connected}
              onValueChange={(value) => setTarget(typeof value === 'number' ? value : value[0])}
              aria-label={`${selected.label} target angle`}
            />
            <div className="slider-limits"><span>{limits[0]}°</span><span>Actual {numberText(degrees(selected.positionRad))}°</span><span>{limits[1]}°</span></div>
            {!state.torqueEnabled && <p className="safety-note">Enable torque to move the selected joint. Targets are initialized from the live pose.</p>}
            {!state.canEnableTorque && state.connected && (
              <p className="safety-note">Waiting for valid feedback: ID {state.unavailableMotorIds.join(', ')}</p>
            )}
            {controlError && <p className="control-error">{controlError}</p>}
          </div>

          <div className="metrics-grid">
            <Metric label="Position" value={`${numberText(degrees(selected.positionRad), 2)}°`} />
            <Metric label="Velocity" value={`${numberText(selected.velocityRadS, 2)} rad/s`} />
            <Metric label="Torque" value={`${numberText(selected.torqueNm, 2)} Nm`} />
            <Metric label="MOS temp" value={`${numberText(selected.tempMosC)} °C`} />
            <Metric label="Bus voltage" value={`${numberText(selected.busVoltageV)} V`} />
            <Metric label="CAN age" value={`${numberText(selected.ageMs, 2)} ms`} />
          </div>

          <div className="state-footer">
            <div><span className={selected.faultCode ? 'state-dot fault' : state.connected ? 'state-dot online' : 'state-dot'} />{selected.state}</div>
            <span>RX {state.stats.receivedPackets.toLocaleString()} · Lost {state.stats.droppedPackets.toLocaleString()}</span>
          </div>
        </aside>
      </section>
    </main>
  );
}
