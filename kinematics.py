import numpy as np
from robot_config import L1, L2, L3, L4, L5

def calculate_fk_transforms(joint_angles):
    """
    Calculates 4x4 Transformation Matrices relative to base frame (T1 through T6).
    """
    t1, t2, t3, t4, t5, t6 = joint_angles
    c = np.cos(joint_angles)
    s = np.sin(joint_angles)

    # DH Parameters / Local Transforms
    A1 = np.array([[c[0], -s[0], 0, 0], [s[0], c[0], 0, 0], [0, 0, 1, L1], [0, 0, 0, 1]])
    A2 = np.array([[c[1], 0, s[1], 0], [s[1], 0, -c[1], 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    A3 = np.array([[c[2], -s[2], 0, L2*c[2]], [s[2], c[2], 0, L2*s[2]], [0, 0, 1, 0], [0, 0, 0, 1]])
    A4 = np.array([[c[3], 0, s[3], 0], [s[3], 0, -c[3], 0], [0, 1, 0, L3], [0, 0, 0, 1]])
    A5 = np.array([[c[4], 0, s[4], L4*c[4]], [s[4], 0, -c[4], L4*s[4]], [0, 1, 0, 0], [0, 0, 0, 1]])
    A6 = np.array([[c[5], -s[5], 0, 0], [s[5], c[5], 0, 0], [0, 0, 1, L5], [0, 0, 0, 1]])

    # Global Transforms (Base to Link)
    T1 = A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5
    T6 = T5 @ A6
    
    return [T1, T2, T3, T4, T5, T6]

def extract_joint_positions(joint_angles):
    """
    Returns XYZ coordinates of every joint (including base at 0,0,0) and end-effector.
    """
    transforms = calculate_fk_transforms(joint_angles)
    p0 = np.array([0.0, 0.0, 0.0])
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