import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random

# --- 1. CONFIGURATION & ROBOT PARAMETERS ---
# Dimensions in mm
L1, L2, L3, L4, L5 = 200.0, 250.0, 200.0, 200.0, 100.0
JOINT_LIMITS = [-np.pi, np.pi]
RRT_STEP_SIZE = 0.15
DT = 0.1  # Time step for simulation (seconds)

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- 2. MATH & KINEMATICS CLASSES ---

def get_transforms(theta):
    """Returns a list of 4x4 transformation matrices for each joint."""
    t1, t2, t3, t4, t5, t6 = theta
    c = np.cos(theta)
    s = np.sin(theta)

    # DH Parameters (Standard)
    # Alpha = [pi/2, 0, pi/2, -pi/2, pi/2, 0] approx for standard 6-DOF
    # This is a simplified DH matching the previous logic
    matrices = [
        np.array([[c[0], -s[0], 0, 0], [s[0], c[0], 0, 0], [0, 0, 1, L1], [0, 0, 0, 1]]),
        np.array([[c[1], 0, s[1], 0], [s[1], 0, -c[1], 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        np.array([[c[2], -s[2], 0, L2*c[2]], [s[2], c[2], 0, L2*s[2]], [0, 0, 1, 0], [0, 0, 0, 1]]),
        np.array([[c[3], 0, s[3], 0], [s[3], 0, -c[3], 0], [0, 1, 0, L3], [0, 0, 0, 1]]),
        np.array([[c[4], 0, s[4], L4*c[4]], [s[4], 0, -c[4], L4*s[4]], [0, 1, 0, 0], [0, 0, 0, 1]]),
        np.array([[c[5], -s[5], 0, 0], [s[5], c[5], 0, 0], [0, 0, 1, L5], [0, 0, 0, 1]])
    ]

    T = np.eye(4)
    transforms = [T]
    for M in matrices:
        T = T @ M
        transforms.append(T)
    return transforms

def get_fk_positions(theta):
    transforms = get_transforms(theta)
    return [T[0:3, 3] for T in transforms]

class GeometricObject:
    def __init__(self, shape_type, x, y, z, width, height, length, name="Obj"):
        self.shape_type = shape_type
        self.x_min, self.x_max = sorted([x, x + width])
        self.y_min, self.y_max = sorted([y, y + length])
        self.z_min, self.z_max = sorted([z, z + height])
        self.y_center = y
        self.z_center = z
        self.outer_r_sq = (width/2)**2
        self.inner_r_sq = (height/2)**2
        self.name = name

    def contains(self, p):
        px, py, pz = p
        if self.shape_type == 'block':
            return (self.x_min <= px <= self.x_max) and \
                   (self.y_min <= py <= self.y_max) and \
                   (self.z_min <= pz <= self.z_max)
        elif self.shape_type == 'hollow_circle':
            # Tube aligned along X-axis
            if not (self.x_min <= px <= self.x_max): return False
            dist_sq = (py - self.y_center)**2 + (pz - self.z_center)**2
            return self.inner_r_sq <= dist_sq <= self.outer_r_sq
        return False

def check_collision(theta, shapes, step_len=50.0):
    joints = get_fk_positions(theta)
    for i in range(len(joints) - 1):
        start, end = joints[i], joints[i+1]
        vec = end - start
        length = np.linalg.norm(vec)
        if length == 0: continue
        steps = int(np.ceil(length / step_len))
        for s in range(steps + 1):
            p = start + vec * (s / steps)
            for shape in shapes:
                if shape.contains(p): return True
    return False

# --- 3. RRT PATH PLANNING ---

class Node:
    def __init__(self, config, parent=None):
        self.config = np.array(config)
        self.parent = parent

class RRTConnectPlanner:
    def __init__(self, start_conf, goal_conf, shapes):
        self.start_node = Node(start_conf)
        self.goal_node = Node(goal_conf)
        self.shapes = shapes
        self.trees = [[self.start_node], [self.goal_node]]

    def get_nearest(self, tree, target_conf):
        dists = [np.linalg.norm(n.config - target_conf) for n in tree]
        return tree[np.argmin(dists)]

    def extend(self, tree, target_conf):
        nearest = self.get_nearest(tree, target_conf)
        diff = target_conf - nearest.config
        dist = np.linalg.norm(diff)
        if dist > RRT_STEP_SIZE:
            new_conf = nearest.config + (diff / dist) * RRT_STEP_SIZE
        else:
            new_conf = target_conf
        
        if check_collision(new_conf, self.shapes): return "Trapped", None
        new_node = Node(new_conf, parent=nearest)
        tree.append(new_node)
        
        if np.linalg.norm(new_conf - target_conf) < 1e-3: return "Reached", new_node
        return "Advanced", new_node

    def plan(self, max_iter=2000):
        idx_a, idx_b = 0, 1
        for i in range(max_iter):
            q_rand = np.random.uniform(JOINT_LIMITS[0], JOINT_LIMITS[1], 6)
            status_a, node_a = self.extend(self.trees[idx_a], q_rand)
            if status_a != "Trapped":
                status_b, last_node = "Advanced", None
                while status_b == "Advanced":
                    status_b, last_node = self.extend(self.trees[idx_b], node_a.config)
                    if status_b == "Reached":
                        return self.reconstruct(node_a, last_node, idx_a)
            idx_a, idx_b = idx_b, idx_a
        return None

    def reconstruct(self, node_a, node_b, tree_a_is_start):
        path_a = []
        curr = node_a
        while curr: path_a.append(curr.config); curr = curr.parent
        path_a.reverse()
        path_b = []
        curr = node_b
        while curr: path_b.append(curr.config); curr = curr.parent
        return np.array(path_a + path_b) if tree_a_is_start == 0 else np.array(path_b[::-1] + path_a[::-1])

# --- 4. DATA GENERATION & METRICS ---

# Environment
floor = GeometricObject('block', x=-400, y=-500, z=0, width=700, height=-100, length=1000, name="floor")
hood = GeometricObject('tube', x=300, y=0, z=790, width=1000, height=1010, length=300, name="hood")
wall =  GeometricObject('block', x=300, y=-500, z=0, width=300, height=790, length=1000, name="wall")
partition = GeometricObject('block', x=300, y=-257.5, z=790, width=300, height=150, length=15, name="partition")

# floor = GeometricObject('block', x=-500, y=-500, z=-10, width=1500, height=1000, length=10, name="Floor")
# obstacle = GeometricObject('block', x=200, y=-200, z=0, width=100, height=400, length=300, name="Wall")
shapes = [floor, wall, partition, hood]

# Define Trajectory
start_conf = np.zeros(6)
goal_conf = np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.0]) # Arbitrary goal

print("Planning Path...")
planner = RRTConnectPlanner(start_conf, goal_conf, shapes)
raw_path = planner.plan()

if raw_path is None:
    print("Failed to find path. Using fallback linear path (may clip obstacles).")
    raw_path = np.linspace(start_conf, goal_conf, 20)

# Interpolate path to time
steps_per_waypoint = 5
traj_theta = []
for i in range(len(raw_path)-1):
    segment = np.linspace(raw_path[i], raw_path[i+1], steps_per_waypoint)
    traj_theta.extend(segment)
traj_theta = np.array(traj_theta)

# Calculate Metrics
time = np.arange(len(traj_theta)) * DT
ee_positions = []
ee_angles = []
joint_vels = np.zeros_like(traj_theta)

# Surface Normal (e.g., Vertical Z-axis for the floor)
surface_normal = np.array([0, 0, 1]) 

for i, theta in enumerate(traj_theta):
    # Position
    transforms = get_transforms(theta)
    pos = transforms[-1][0:3, 3]
    ee_positions.append(pos)
    
    # Orientation Angle (Dot product of End Effector Z-axis vs Surface Normal)
    ee_z_axis = transforms[-1][0:3, 2] # The approach vector of the tool
    dot_prod = np.dot(ee_z_axis, surface_normal)
    # Clip for safety (acos domain)
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    angle = np.arccos(dot_prod)
    ee_angles.append(angle)
    
    # Velocity (Finite Difference)
    if i > 0:
        joint_vels[i] = (traj_theta[i] - traj_theta[i-1]) / DT

ee_positions = np.array(ee_positions)
ee_angles = np.array(ee_angles)

# --- 5. PLOTTING THE 4 DATA GRAPHS ---

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 1. EE Position vs Time
axs[0, 0].plot(time, ee_positions[:, 0], label='X')
axs[0, 0].plot(time, ee_positions[:, 1], label='Y')
axs[0, 0].plot(time, ee_positions[:, 2], label='Z')
axs[0, 0].set_title('End-Effector Position (mm)')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. Angle vs Time
axs[0, 1].plot(time, np.degrees(ee_angles), color='purple')
axs[0, 1].set_title('Tool Angle Relative to Surface Normal (deg)')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].grid(True)

# 3. Joint Angles vs Time
for j in range(6):
    axs[1, 0].plot(time, traj_theta[:, j], label=f'J{j+1}')
axs[1, 0].set_title('Joint Angles (rad)')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].legend(ncol=2, fontsize='small')
axs[1, 0].grid(True)

# 4. Joint Velocities vs Time
for j in range(6):
    axs[1, 1].plot(time, joint_vels[:, j], label=f'v{j+1}')
axs[1, 1].set_title('Joint Velocities (rad/s)')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].grid(True)

plt.show()

# --- 6. 3D SIMULATION ---

fig_sim = plt.figure(figsize=(10, 8))
ax_sim = fig_sim.add_subplot(111, projection='3d')

def update_sim(frame):
    ax_sim.clear()
    
    # Settings
    ax_sim.set_xlim(-600, 600)
    ax_sim.set_ylim(-600, 600)
    ax_sim.set_zlim(0, 1000)
    ax_sim.set_title(f"Time: {frame*DT:.2f}s")
    ax_sim.set_xlabel('X')
    ax_sim.set_ylabel('Y')
    ax_sim.set_zlabel('Z')
    
    # Draw Obstacles
    for s in shapes:
        if s.shape_type == 'block':
            dx, dy, dz = s.x_max-s.x_min, s.y_max-s.y_min, s.z_max-s.z_min
            ax_sim.bar3d(s.x_min, s.y_min, s.z_min, dx, dy, dz, color='red', alpha=0.1)
            
        elif s.shape_type == 'tube' or s.shape_type == 'hollow_circle':
            # Parametric generation of a tube aligned along X-axis
            n_samples = 20
            # Outer surface radius
            r_outer = np.sqrt(s.outer_r_sq)
            # Length along X
            x_range = np.linspace(s.x_min, s.x_max, 2)
            theta_range = np.linspace(0, 2*np.pi, n_samples)
            
            X, T = np.meshgrid(x_range, theta_range)
            # Center the circle on y_center and z_center
            Y = r_outer * np.cos(T) + s.y_center
            Z = r_outer * np.sin(T) + s.z_center
            
            # Plot the outer shell
            ax_sim.plot_wireframe(X, Y, Z, color='red', alpha=0.3)
            
            # Optional: Plot the inner shell if it's a "hollow" tube
            r_inner = np.sqrt(s.inner_r_sq)
            if r_inner > 0:
                Y_in = r_inner * np.cos(T) + s.y_center
                Z_in = r_inner * np.sin(T) + s.z_center
                ax_sim.plot_wireframe(X, Y_in, Z_in, color='darkred', alpha=0.2)

    # Draw Robot
    current_theta = traj_theta[frame]
    joint_pos = get_fk_positions(current_theta)
    
    xs = [p[0] for p in joint_pos]
    ys = [p[1] for p in joint_pos]
    zs = [p[2] for p in joint_pos]
    
    # Plot Links and Joints
    ax_sim.plot(xs, ys, zs, 'o-', linewidth=4, markersize=8, color='blue', label='Robot Arm')
    ax_sim.scatter([0], [0], [0], color='black', s=100) # Base


print("Starting Animation...")
ani = animation.FuncAnimation(fig_sim, update_sim, frames=len(traj_theta), interval=50, repeat=True)
plt.show()