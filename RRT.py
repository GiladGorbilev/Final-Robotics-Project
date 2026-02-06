import numpy as np
import random
from main import *

# --- (Keep your existing Robot Dimensions, get_transforms, pos, newtheta, GeometricObject, check_robot_collision here) ---

# --- RRT Constants ---
JOINT_LIMITS = [-np.pi, np.pi] # Limits for valid joint angles
MAX_ITER = 5000                # Max attempts to find a path
RRT_STEP_SIZE = 0.1            # Max rotation (radians) per step
GOAL_BIAS = 0.1                # 10% chance to sample the goal directly

class RRTNode:
    """Helper class to store the RRT tree structure."""
    def __init__(self, theta, parent=None):
        self.theta = theta
        self.parent = parent  # Reference to parent node object

def steer(from_theta, to_theta, step_size):
    """
    Moves from 'from_theta' toward 'to_theta' by a max distance of 'step_size'.
    """
    diff = to_theta - from_theta
    distance = np.linalg.norm(diff)
    
    # If the target is closer than the step size, go straight there
    if distance < step_size:
        return to_theta
    
    # Otherwise, move incrementally in that direction
    return from_theta + (diff / distance) * step_size

def rrt_path_planner(start_theta, target_pos, shapes):
    """
    Generates a path of theta configurations from start_theta to target_pos.
    """
    start_theta = np.array(start_theta)
    
    # 1. IK PRE-CALCULATION
    # RRT works in Joint Space (Theta), but your target is in Workspace (XYZ).
    # We must first find a valid 'target_theta' that reaches the XYZ coords.
    print("Calculating Inverse Kinematics for target...")
    target_theta = np.array(start_theta)
    
    # Use your existing Jacobian solver to find the goal configuration
    for _ in range(500):
        target_theta, err = compute_ik_step(target_theta, target_pos)
        if err < 0.05: break
        
    if err > 0.05:
        print("Error: Target Position is out of reach.")
        return None
            
    # Check if the goal itself is safe
    if check_robot_collision(target_theta, shapes):
        print("Target position is invalid (inside obstacle).")
        return None

    # 2. INITIALIZE TREE
    start_node = RRTNode(start_theta)
    nodes = [start_node]
    
    print("Starting RRT search...")

    for i in range(MAX_ITER):
        # A. SAMPLE
        # With 10% probability, sample the goal to speed up convergence
        if random.random() < GOAL_BIAS:
            q_rand = target_theta
        else:
            # Random configuration within limits
            q_rand = np.random.uniform(JOINT_LIMITS[0], JOINT_LIMITS[1], size=6)

        # B. NEAREST NEIGHBOR
        # Find the node in the tree closest to q_rand
        nearest_node = nodes[0]
        min_dist = float('inf')
        
        for node in nodes:
            d = np.linalg.norm(node.theta - q_rand)
            if d < min_dist:
                min_dist = d
                nearest_node = node

        # C. STEER
        # Move from nearest_node toward q_rand by step_size
        q_new = steer(nearest_node.theta, q_rand, RRT_STEP_SIZE)

        # D. COLLISION CHECK
        if not check_robot_collision(q_new, shapes):
            # Safe! Add to tree
            new_node = RRTNode(q_new, parent=nearest_node)
            nodes.append(new_node)
            
            # E. CHECK FOR SUCCESS
            # If we are very close to the target theta
            if np.linalg.norm(q_new - target_theta) < RRT_STEP_SIZE:
                print(f"Path found in {i} iterations!")
                
                # Backtrack to reconstruct the path
                path = []
                curr = new_node
                while curr is not None:
                    path.append(curr.theta)
                    curr = curr.parent
                return np.array(path[::-1]) # Return reversed (Start -> End)

    print("RRT failed to find a path.")
    return None

# --- Usage Example ---

# Define Environment
floor = GeometricObject('block', x=-400, y=-500, z=0, width=1000, height=-100, length=700)
hood = GeometricObject('tube', x=300, y=0, z=790, width=500, height=510, length=300)
wall =  GeometricObject('block', x=300, y=-500, z=0, width=1000, height=790, length=300)
partition = GeometricObject('block', x=300, y=-257.5, z=790, width=15, height=150, length=300)

env_shapes = [floor, hood]

# Define Goal
start_conf = np.zeros(6)
target_xyz = [2.0, 2.0, 2.0] # Adjust based on your scale (your L1 is 1.0, so this is reasonable)

# Run RRT
path = rrt_path_planner(start_conf, target_xyz, env_shapes)

if path is not None:
    print(f"Generated path with {len(path)} steps.")
    # You can now feed 'path' into your visualization or controller