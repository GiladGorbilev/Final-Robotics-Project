import numpy as np

# --- Robot Dimensions (mm) ---
L1, L2, L3, L4, L5 = 200.0, 250.0, 200.0, 200.0, 100.0

# --- Simulation Constants ---
JOINT_LIMITS = [-np.pi, np.pi]
RRT_STEP_SIZE = 0.15
DT = 0.1  # Time step for simulation