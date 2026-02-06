import numpy as np
import math

# --- Robot Dimensions (Global) ---
L1 = 1.0
L2 = 1.0
L3 = 1.0
L4 = 1.0
L5 = 0.5
dt = 0.01

def get_transforms(theta): # Helper to get all T matrices Alon Bicknell
    t1, t2, t3, t4, t5, t6 = theta
    
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Transformation Matrices (A1 to A6)
    A1 = np.array([[c[0], -s[0], 0, 0], [s[0], c[0], 0, 0], [0, 0, 1, L1], [0, 0, 0, 1]])
    A2 = np.array([[c[1], 0, s[1], 0], [s[1], 0, -c[1], 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    A3 = np.array([[c[2], -s[2], 0, L2*c[2]], [s[2], c[2], 0, L2*s[2]], [0, 0, 1, 0], [0, 0, 0, 1]])
    A4 = np.array([[c[3], 0, s[3], 0], [s[3], 0, -c[3], 0], [0, 1, 0, L3], [0, 0, 0, 1]])
    A5 = np.array([[c[4], 0, s[4], L4*c[4]], [s[4], 0, -c[4], L4*s[4]], [0, 1, 0, 0], [0, 0, 0, 1]])
    A6 = np.array([[c[5], -s[5], 0, 0], [s[5], c[5], 0, 0], [0, 0, 1, L5], [0, 0, 0, 1]])

    # Forward Kinematics
    T1 = A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5
    T6 = T5 @ A6
    
    return [T1, T2, T3, T4, T5, T6]

def pos(theta): # Returns all joint positions Alon Bicknell
    """
    Returns a (7, 3) array containing the [x, y, z] coordinates of:
    Base -> Joint1 -> Joint2 -> Joint3 -> Joint4 -> Joint5 -> EndEffector
    """
    T = get_transforms(theta)
    
    # Robot Base at (0,0,0)
    p0 = np.array([0, 0, 0])
    
    # Extract positions from the transformation matrices
    p1 = T[0][0:3, 3]
    p2 = T[1][0:3, 3]
    p3 = T[2][0:3, 3]
    p4 = T[3][0:3, 3]
    p5 = T[4][0:3, 3]
    p6 = T[5][0:3, 3]
    
    return np.array([p0, p1, p2, p3, p4, p5, p6])

def newtheta(theta, target, dt=0.01): # Alon Bicknell
    # 1. Get current End Effector Position
    all_positions = pos(theta)
    current_pos = all_positions[-1] # The last point is p6
    
    # 2. Compute Error
    error_vec = np.array(target) - current_pos
    error_dist = np.linalg.norm(error_vec)
    
    # 3. Compute Jacobian (Numerical or Geometric approach)
    # We need T matrices for the z-vectors to build J
    T = get_transforms(theta)
    J_pos = np.zeros((6, 6))
    
    # T_prev starts as Identity (base frame)
    T_prev = np.eye(4)
    
    for i in range(6):
        z_prev = T_prev[0:3, 2] # Z-axis of previous joint
        p_prev = T_prev[0:3, 3] # Position of previous joint
        
        # Cross product for linear velocity influence
        J_pos[:, i] = np.cross(z_prev, current_pos - p_prev)
        
        # Update T_prev to the current joint's transform
        T_prev = T[i]

    # 4. Apply P-Controller to get velocity
    Kp = 5.0
    desired_vel = Kp * error_vec
    
    # 5. Inverse Kinematics (theta_dot = pinv(J) * v)
    theta_dot = np.linalg.pinv(J_pos) @ desired_vel
    theta_dot = np.clip(theta_dot, -10.0, 10.0) # Safety limit
    
    return theta + theta_dot * dt, error_dist

def thetas_path(start_theta, target, dt=0.01): # Alon Bicknell
    theta_list = []
    current_theta = np.array(start_theta)
    
    for i in range(2000):
        theta_list.append(current_theta)
        
        current_theta, error = newtheta(current_theta, target, dt)
        
        if error < 0.01:
            print(f"Converged in {i} steps")
            break
            
    return np.array(theta_list)

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
            # The block starts at x,y,z and extends positively
            in_x = self.x <= px <= (self.x + self.width)
            in_y = self.y <= py <= (self.y + self.height)
            in_z = self.z <= pz <= (self.z + self.length)
            
            return in_x and in_y and in_z

        # 2. TUBE LOGIC (Center-based XY, Start-based Z)
        elif self.shape_type == 'hollow_circle':
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
    joints = pos(theta) 
    
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

# D. Integration with your 'thetas_path' loop (Concept)
# You can update your path planner to stop if a collision is detected:
def safe_thetas_path(start_theta, target, shapes):
    current_theta = np.array(start_theta)
    for i in range(2000):
        # 1. Calculate next step
        next_theta, err = newtheta(current_theta, target)
        
        # 2. PREDICT: Will the next step crash?
        if check_robot_collision(next_theta, shapes):
            print("Emergency Stop: Next move causes collision.")
            break
            
        current_theta = next_theta

print(safe_thetas_path(test_theta, [0, 0, 0,0 ,0 ,0], shapes=env_shapes))
