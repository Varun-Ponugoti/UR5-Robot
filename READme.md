# UR5 Robot Simulator

An interactive 3D simulator for the UR5 6-DOF robotic arm, built with Python and Matplotlib. Supports forward kinematics control via rotary knobs and trajectory planning with real-time singularity detection.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![Matplotlib](https://img.shields.io/badge/matplotlib-required-orange) ![NumPy](https://img.shields.io/badge/numpy-required-orange)

---

## Features

- **Forward Kinematics** — drag rotary knobs to set each of the 6 joint angles and watch the arm update in real time
- **Trajectory Planning** — move the end effector along a Line, Square, or Circle path using damped least-squares inverse kinematics
- **Singularity Detection** — live Jacobian analysis with manipulability index and condition number; arm turns red and trajectory halts at singularities
- **Coordinate Frames** — RGB axes drawn at every joint showing orientation
- **Trajectory History** — completed paths are drawn on the 3D plot for reference

---

## Project Structure

```
ur5sim/
├── simulate.py         # Entry point — run this
├── ur5_visualizer.py   # UR5Visualizer class (Matplotlib UI)
├── ur5_robot.py        # UR5Robot class (kinematics & motion planning)
└── knob.py             # Knob widget class (interactive rotary control)
```

---

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip3 install numpy matplotlib
```

---

## Usage

```bash
cd ur5sim
python3 simulate.py
```

---

## Controls

### Forward Kinematics Mode
Six rotary knobs control joints θ1–θ6. Click and drag within a knob to rotate the joint. The current angle (in degrees) is displayed above each knob.

### Trajectory Mode
Switch modes using the **Forward K / Trajectory** radio buttons.

| Control | Description |
|---|---|
| Tgt X / Y / Z sliders | Set the target end-effector position |
| Line / Square / Circle | Choose trajectory shape |
| XY / XZ / YZ | Choose the plane for Square and Circle paths |
| Execute | Run the trajectory animation |
| Reset | Return the arm to the home position |

---

## Kinematics

The robot is modelled using the standard **Denavit-Hartenberg (DH) parameters** for the Universal Robots UR5:

| Joint | a (m) | α (rad) | d (m) |
|---|---|---|---|
| 1 | 0 | π/2 | 0.0892 |
| 2 | −0.425 | 0 | 0 |
| 3 | −0.392 | 0 | 0 |
| 4 | 0 | π/2 | 0.1093 |
| 5 | 0 | −π/2 | 0.09475 |
| 6 | 0 | 0 | 0.0825 |

Inverse kinematics is solved iteratively using the **damped least-squares (DLS)** method to avoid instability near singularities.

### Singularity Detection

At each step, the Jacobian is evaluated and three metrics are checked:

- **Manipulability index** `√det(J·Jᵀ)` — below `1e-3` flags a singularity
- **Condition number** of the position Jacobian — above `1e4` flags a singularity
- **Determinant** of `J·Jᵀ` — below `1e-4` flags a singularity

If any condition is met, the arm highlights red and active trajectories are halted.
