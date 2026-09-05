"""Constants shared by all Pochi simulation backends.

Geometry is taken from the Onshape CAD export; see
``scripts/glb_to_mjcf.py`` and ``assets/pochi/README.md``.
"""

LEGS = ("FL", "FR", "RL", "RR")
JOINT_KINDS = ("hip_roll", "hip_pitch", "knee")
JOINT_NAMES = tuple(f"{leg}_{kind}" for leg in LEGS for kind in JOINT_KINDS)

# --- RobStride RS02 actuator (datasheet) --------------------------------------
# Every joint is an RS02 quasi-direct-drive module.  Twelve of them dominate the
# mass budget, and their torque envelope is what the policy has to live inside,
# so these numbers are shared by every backend rather than hidden in the MJCF.
RS02_MASS_KG = 0.405  # 405 g +/- 5 g; the RobStride repo quotes 380 g +/- 3 g
RS02_PEAK_TORQUE_NM = 17.0  # stall torque, only available near zero speed
RS02_RATED_TORQUE_NM = 7.0  # continuous rating, quoted at 100 rpm
RS02_NO_LOAD_SPEED_RAD_S = 42.9  # 410 rpm +/- 10% at the output shaft
RS02_GEAR_RATIO = 7.75
RS02_REFLECTED_INERTIA = 0.01  # rotor inertia through the gearbox [kg m^2]

# CAD-derived link geometry [m].  The hip roll and hip pitch axes intersect, so
# both hip joints share one origin.
HIP_OFFSET_X = 0.2545  # base centre -> hip joint, fore/aft
HIP_OFFSET_Y = 0.07  # base centre -> hip joint, lateral
THIGH_LENGTH = 0.200  # hip pitch axis -> knee axis
SHANK_LENGTH = 0.225  # knee axis -> foot tip
FOOT_OFFSET_Y = 0.098  # leg plane -> foot tip, lateral (outboard)

# --- Motor mounting directions ------------------------------------------------
# Which way each real RS02 is bolted in, as a sign against the *canonical* leg
# frame (all four legs identical: hip roll about +x, hip pitch and knee about
# +y, the frame `pochi_rl.control.leg_kinematics` solves in).  The modules are
# fitted in a rotationally symmetric pattern rather than a mirrored one, so the
# signs come out diagonal: FL matches RR, FR matches RL.
#
# These are the single source of truth for the convention, and the MJCF is
# generated to match (`pochi_rl.cad.convert` negates the joint axis wherever the
# sign is -1).  The point is that a joint angle here means the same number the
# Teensy reports and accepts for that motor, so nothing has to convert between
# simulation and hardware.
#
# Verified on the physical robot by driving one joint at a time; the hardware
# side of the same fact is `VIEWER_REVERSED_MOTOR_IDS` in
# `pochi_hardware/web/app/page.tsx`.  If CAD ever disagrees, the robot wins.
MOTOR_SIGN = {
  "FL_hip_roll": -1.0,
  "FL_hip_pitch": -1.0,
  "FL_knee": -1.0,
  "FR_hip_roll": -1.0,
  "FR_hip_pitch": 1.0,
  "FR_knee": 1.0,
  "RL_hip_roll": 1.0,
  "RL_hip_pitch": -1.0,
  "RL_knee": -1.0,
  "RR_hip_roll": 1.0,
  "RR_hip_pitch": 1.0,
  "RR_knee": 1.0,
}

# All four legs are kinematically identical, so the stance is a free choice.
# The front/rear mirrored ("M") layout is how the CAD is assembled and it
# balances the robot: each foot carries a quarter of the weight and the COM
# stays over the base centre.  The uniform layout shifts the COM 41 mm back.
#
# Written in motor coordinates, so the front/rear mirror is not visible as a
# sign flip: MOTOR_SIGN already absorbs it, and the diagonal mounting is what
# makes FL equal RR and FR equal RL below.
DEFAULT_JOINT_POS = {
  "FL_hip_roll": 0.0,
  "FL_hip_pitch": -0.8,
  "FL_knee": 1.5,
  "FR_hip_roll": 0.0,
  "FR_hip_pitch": 0.8,
  "FR_knee": -1.5,
  "RL_hip_roll": 0.0,
  "RL_hip_pitch": 0.8,
  "RL_knee": -1.5,
  "RR_hip_roll": 0.0,
  "RR_hip_pitch": -0.8,
  "RR_knee": 1.5,
}

# --- Torso ballast ------------------------------------------------------------
# Added weight carried on the base link.  The bare CAD torso is light for the
# machine's size, so this stands in for the battery/electronics/ballast the real
# robot carries.  Modelled as a steel block: the density fixes its size and
# therefore its rotational inertia, which a bare point mass would leave at zero
# and make the torso far too easy to spin.  Set PAYLOAD_MASS_KG to 0 to drop it.
PAYLOAD_MASS_KG = 3.0
PAYLOAD_POS = (0.0, 0.0, 0.0)  # in the base link frame, i.e. the hip centre
PAYLOAD_DENSITY = 7850.0  # steel [kg/m^3]

# Base height with the feet on the ground at DEFAULT_JOINT_POS.  Measured off
# the generated MJCF's foot-pad collision meshes; test_mjcf_loads pins it.
NOMINAL_BASE_HEIGHT = 0.3173

FOOT_BODIES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in LEGS)
FOOT_SITES = tuple(f"{leg}_foot_site" for leg in LEGS)
THIGH_GEOMS = tuple(f"{leg}_thigh_collision" for leg in LEGS)
SHANK_GEOMS = tuple(f"{leg}_shank_collision" for leg in LEGS)
BASE_BODY = "base_link"
IMU_SITE = "imu"
