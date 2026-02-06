import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from robot_config import DT
from kinematics import calculate_fk_transforms, extract_joint_positions
from geometry import GeometricObject
from planner import RRTConnectPlanner

# --- 1. ENVIRONMENT SETUP ---
# Using definitions from Snippet 2 where applicable, mapped to GeometricObject
floor = GeometricObject('block', x=-400, y=-500, z=0, width=1000, height=-100, length=700, name="floor")
# Note: For the tube, we use the specific tube definition logic from Snippet 2
# (x is start X, width=OuterDia, height=InnerDia, length=Length along X)
hood = GeometricObject('tube', x=300, y=0, z=790, width=1000, height=1010, length=300, name="hood")
wall =  GeometricObject('block', x=300, y=-500, z=0, width=1000, height=790, length=300, name="wall")
partition = GeometricObject('block', x=300, y=-257.5, z=790, width=15, height=150, length=300, name="partition")

shapes = [floor, wall, partition, hood]

# --- 2. PATH PLANNING ---
start_conf = np.zeros(6)
goal_conf = np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.0])

print("Planning Path...")
planner = RRTConnectPlanner(start_conf, goal_conf, shapes)
raw_path = planner.plan()

if raw_path is None:
    print("Failed to find path. Using fallback linear path.")
    raw_path = np.linspace(start_conf, goal_conf, 20)
else:
    print(f"Path found with {len(raw_path)} waypoints.")

# Interpolate path
steps_per_waypoint = 5
traj_theta = []
for i in range(len(raw_path)-1):
    segment = np.linspace(raw_path[i], raw_path[i+1], steps_per_waypoint)
    traj_theta.extend(segment)
traj_theta = np.array(traj_theta)

# --- 3. METRICS ---
time = np.arange(len(traj_theta)) * DT
ee_positions = []
ee_angles = []
joint_vels = np.zeros_like(traj_theta)
surface_normal = np.array([0, 0, 1]) 

for i, theta in enumerate(traj_theta):
    transforms = calculate_fk_transforms(theta)
    pos = transforms[-1][0:3, 3]
    ee_positions.append(pos)
    
    # Angle metric
    ee_z_axis = transforms[-1][0:3, 2]
    dot_prod = np.clip(np.dot(ee_z_axis, surface_normal), -1.0, 1.0)
    ee_angles.append(np.arccos(dot_prod))
    
    if i > 0:
        joint_vels[i] = (traj_theta[i] - traj_theta[i-1]) / DT

ee_positions = np.array(ee_positions)
ee_angles = np.array(ee_angles)

# --- 4. VISUALIZATION ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Graphs
axs[0, 0].plot(time, ee_positions[:, 0], label='X')
axs[0, 0].plot(time, ee_positions[:, 1], label='Y')
axs[0, 0].plot(time, ee_positions[:, 2], label='Z')
axs[0, 0].set_title('End-Effector Position (mm)')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(time, np.degrees(ee_angles), color='purple')
axs[0, 1].set_title('Tool Angle (deg)')
axs[0, 1].grid(True)

for j in range(6):
    axs[1, 0].plot(time, traj_theta[:, j], label=f'J{j+1}')
axs[1, 0].set_title('Joint Angles (rad)')
axs[1, 0].grid(True)

for j in range(6):
    axs[1, 1].plot(time, joint_vels[:, j], label=f'v{j+1}')
axs[1, 1].set_title('Joint Velocities (rad/s)')
axs[1, 1].grid(True)

plt.show()

# 3D Animation
fig_sim = plt.figure(figsize=(10, 8))
ax_sim = fig_sim.add_subplot(111, projection='3d')

def update_sim(frame):
    ax_sim.clear()
    ax_sim.set_xlim(-600, 600)
    ax_sim.set_ylim(-600, 600)
    ax_sim.set_zlim(0, 1000)
    ax_sim.set_title(f"Time: {frame*DT:.2f}s")
    
    # Draw Shapes
    for s in shapes:
        if s.shape_type == 'block':
            dx = s.x_max - s.x_min
            dy = s.y_max - s.y_min
            dz = s.z_max - s.z_min
            # Ensure positive dimensions for bar3d
            ax_sim.bar3d(s.x_min, s.y_min, s.z_min, dx, dy, dz, color='red', alpha=0.1)
            
        elif s.shape_type == 'tube':
            # Parametric tube generation for visual
            n_samples = 20
            r_outer = s.width / 2
            x_range = np.linspace(s.x, s.x + s.length, 2)
            theta_range = np.linspace(0, 2*np.pi, n_samples)
            X, T = np.meshgrid(x_range, theta_range)
            Y = r_outer * np.cos(T) + s.y
            Z = r_outer * np.sin(T) + s.z
            ax_sim.plot_wireframe(X, Y, Z, color='red', alpha=0.3)

    # Draw Robot
    current_theta = traj_theta[frame]
    joint_pos = extract_joint_positions(current_theta)
    
    xs = joint_pos[:, 0]
    ys = joint_pos[:, 1]
    zs = joint_pos[:, 2]
    
    ax_sim.plot(xs, ys, zs, 'o-', linewidth=4, markersize=8, color='blue')
    ax_sim.scatter([0], [0], [0], color='black', s=100)

print("Starting Animation...")
ani = animation.FuncAnimation(fig_sim, update_sim, frames=len(traj_theta), interval=50, repeat=True)
plt.show()