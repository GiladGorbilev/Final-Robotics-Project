import numpy as np
import math

# --- Robot Dimensions (Global) ---
# Update Robot Dimensions to match the scale of the environment (e.g., mm)
L1, L2, L3, L4, L5 = 200.0, 250.0, 200.0, 200.0, 100.0

def calculate_fk_transforms(joint_angles):
    """
    Calculates the Forward Kinematics (FK).
    Returns a list of 4x4 Transformation Matrices relative to the base frame
    for each link (T1 through T6).
    """
    # Unpack angles
    t1, t2, t3, t4, t5, t6 = joint_angles
    
    # Precompute cosines and sines
    c = np.cos(joint_angles)
    s = np.sin(joint_angles)
    
    # Local Transformation Matrices (DH Parameters)
    # These represent the offset and rotation from one joint to the next.
    A1 = np.array([[c[0], -s[0], 0, 0], [s[0], c[0], 0, 0], [0, 0, 1, L1], [0, 0, 0, 1]])
    A2 = np.array([[c[1], 0, s[1], 0], [s[1], 0, -c[1], 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    A3 = np.array([[c[2], -s[2], 0, L2*c[2]], [s[2], c[2], 0, L2*s[2]], [0, 0, 1, 0], [0, 0, 0, 1]])
    A4 = np.array([[c[3], 0, s[3], 0], [s[3], 0, -c[3], 0], [0, 1, 0, L3], [0, 0, 0, 1]])
    A5 = np.array([[c[4], 0, s[4], L4*c[4]], [s[4], 0, -c[4], L4*s[4]], [0, 1, 0, 0], [0, 0, 0, 1]])
    A6 = np.array([[c[5], -s[5], 0, 0], [s[5], c[5], 0, 0], [0, 0, 1, L5], [0, 0, 0, 1]])

    # Global Transformation Matrices (Base to Link)
    # We multiply them sequentially to get the position relative to the base (0,0,0).
    T1 = A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5
    T6 = T5 @ A6
    
    return [T1, T2, T3, T4, T5, T6]

def extract_joint_positions(joint_angles):
    """
    Extracts the XYZ coordinates of every joint and the end-effector.
    Useful for visualization.
    """
    transforms = calculate_fk_transforms(joint_angles)
    
    # Robot Base at (0,0,0)
    p0 = np.array([0.0, 0.0, 0.0])
    
    # Extract the translation column (first 3 rows, 4th column) from each matrix
    positions = [p0] + [T[0:3, 3] for T in transforms]
    
    return np.array(positions)

def compute_ik_step(current_angles, target_pos, dt=0.01):
    """
    Performs one iteration of Inverse Kinematics using the Jacobian Inverse method.
    Calculates the angular velocity needed to move the end-effector toward the target.
    """
    # 1. Get current End Effector Position (XYZ)
    transforms = calculate_fk_transforms(current_angles)
    current_end_effector_pos = transforms[-1][0:3, 3]
    
    # 2. Compute the Position Error Vector (Direction to target)
    error_vec = np.array(target_pos) - current_end_effector_pos
    error_distance = np.linalg.norm(error_vec)
    
    # 3. Compute the Jacobian Matrix (3x6 for Position only)
    # The Jacobian relates Joint Velocities (dTheta) to End-Effector Velocities (dV)
    J_pos = np.zeros((3, 6))
    
    # Base frame (Identity)
    prev_transform = np.eye(4)
    
    for i in range(6):
        # z_axis: The axis of rotation for the current joint
        z_axis = prev_transform[0:3, 2] 
        # p_joint: The position of the current joint
        p_joint = prev_transform[0:3, 3]
        
        # The geometric Jacobian column for a revolute joint is: Z x (P_end - P_joint)
        J_pos[:, i] = np.cross(z_axis, current_end_effector_pos - p_joint)
        
        # Update transform for the next iteration
        prev_transform = transforms[i]

    # 4. Apply P-Controller (Proportional Control)
    # We want to move the robot with a velocity proportional to the error
    Kp = 5.0
    desired_velocity = Kp * error_vec
    
    # 5. Solve for Joint Velocities (Inverse Kinematics)
    # theta_dot = pseudo_inverse(J) * desired_velocity
    # This solves the equation: J * theta_dot = v
    J_pinv = np.linalg.pinv(J_pos)
    theta_dot = J_pinv @ desired_velocity
    
    # Clip speeds for safety/stability
    theta_dot = np.clip(theta_dot, -10.0, 10.0)
    
    # 6. Integrate to get new angles: new_pos = old_pos + velocity * time
    new_angles = current_angles + theta_dot * dt
    
    return new_angles, error_distance

def solve_trajectory(start_angles, target_pos, dt=0.01, max_steps=2000, tolerance=0.01):
    """
    Iteratively moves the robot until the end-effector reaches the target position.
    """
    path_history = []
    current_angles = np.array(start_angles, dtype=float)
    
    for step in range(max_steps):
        path_history.append(current_angles)
        
        current_angles, error = compute_ik_step(current_angles, target_pos, dt)
        
        if error < tolerance:
            print(f"Converged in {step} steps. Final Error: {error:.4f}")
            break
            
    return np.array(path_history)

class GeometricObject:
    def __init__(self, shape_type, x, y, z, width, height, length):
        self.shape_type = shape_type
        self.x, self.y, self.z = x, y, z
        self.width, self.height, self.length = width, height, length

    def contains_point(self, px, py, pz):
        """
        Checks collision based on the specific coordinate systems:
        - Block: (x,y,z) is the minimum corner.
        - Tube: (x,y) is center, z is the start of the length.
        """
        
        # 1. BLOCK LOGIC (Corner-based)
        if self.shape_type == 'block':
            # Ensure we check from Min to Max, regardless of how width/height were defined
            x_min, x_max = sorted([self.x, self.x + self.width])
            y_min, y_max = sorted([self.y, self.y + self.height])
            z_min, z_max = sorted([self.z, self.z + self.length])

            return (x_min <= px <= x_max) and \
                (y_min <= py <= y_max) and \
                (z_min <= pz <= z_max)

        # 2. TUBE LOGIC (Center-based XY, Start-based Z)
        elif self.shape_type == 'tube':
            # Length Check (along X-axis)
            if not (self.x <= px <= (self.x + self.length)):
                return False

            # Circle Check (in YZ plane)
            # Calculate distance from center (self.y, self.z)
            dy = py - self.y
            dz = pz - self.z
            dist_sq = dy*dy + dz*dz

            # Radii definitions
            # Width = Outer Diameter -> Radius = Width / 2
            # Height = Inner Diameter -> Radius = Height / 2
            r_outer_sq = (self.width / 2) ** 2
            r_inner_sq = (self.height / 2) ** 2

            # Check if point is inside the ring (between inner and outer radius)
            return r_inner_sq <= dist_sq <= r_outer_sq

        return False

# --- Helper to visualize the change ---
# Use this within your existing robot collision loop

# --- 2. The Collision Detection Logic ---

def check_robot_collision(theta, shapes, step_size=0.1):
    """
    Checks if any part of the robot (joints OR links) intersects with any shape.
    
    Args:
        theta: The joint angles.
        shapes: List of GeometricObject.
        step_size: How far apart to check points along the arm links. 
                   Smaller = more accurate but slower.
    
    Returns:
        True if collision detected, False otherwise.
    """
    # 1. Get all joint positions (p0, p1, ... p6) using your existing function
    joints = extract_joint_positions(theta) 
    
    # 2. Iterate through every link (Base->J1, J1->J2, etc.)
    for i in range(len(joints) - 1):
        start_pt = joints[i]
        end_pt = joints[i+1]
        
        # Calculate length of this link
        vec = end_pt - start_pt
        link_len = np.linalg.norm(vec)
        
        # 3. Sample points along the link (Linear Interpolation)
        # We perform 'num_steps' checks along this line segment
        if link_len == 0: continue # Skip zero-length links
        
        num_steps = int(math.ceil(link_len / step_size))
        
        for step in range(num_steps + 1):
            t = step / num_steps # t goes from 0.0 to 1.0
            
            # Interpolated point P = Start + (Vector * t)
            check_p = start_pt + (vec * t)
            
            # 4. Check this specific point against ALL shapes
            px, py, pz = check_p
            for shape in shapes:
                if shape.contains_point(px, py, pz):
                    # Collision Found! 
                    # Optional: Print detailed info
                    # print(f"Collision detected at link {i}->{i+1} on shape {shape.shape_type}")
                    return True

    return False

# --- 3. Example Simulation Integration ---

# A. Create the Environment
# A block blocking the path
floor = GeometricObject('block', x=-400, y=-500, z=0, width=1000, height=-100, length=700)
hood = GeometricObject('tube', x=300, y=0, z=790, width=500, height=510, length=300)
wall =  GeometricObject('block', x=300, y=-500, z=0, width=1000, height=790, length=300)
partition = GeometricObject('block', x=300, y=-257.5, z=790, width=15, height=150, length=300)

env_shapes = [floor, hood]

# B. Define a Test Configuration
test_theta = np.zeros(6) # Robot pointing straight up/out

# C. Check Collision
is_hit = check_robot_collision(test_theta, env_shapes)

if is_hit:
    print("CRITICAL WARNING: Robot path is blocked!")
else:
    print("Path is clear.")
