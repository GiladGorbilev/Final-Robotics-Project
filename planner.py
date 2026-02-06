import numpy as np
import math
from robot_config import JOINT_LIMITS, RRT_STEP_SIZE
from kinematics import extract_joint_positions

# --- Collision Logic (Adapted from Snippet 2) ---
def check_robot_collision(theta, shapes, step_size=50.0):
    """
    Checks if any part of the robot intersects with any shape.
    """
    joints = extract_joint_positions(theta) 
    
    for i in range(len(joints) - 1):
        start_pt = joints[i]
        end_pt = joints[i+1]
        
        vec = end_pt - start_pt
        link_len = np.linalg.norm(vec)
        
        if link_len == 0: continue
        
        num_steps = int(math.ceil(link_len / step_size))
        
        for step in range(num_steps + 1):
            t = step / num_steps
            check_p = start_pt + (vec * t)
            px, py, pz = check_p
            
            for shape in shapes:
                if shape.contains_point(px, py, pz):
                    return True
    return False

# --- RRT Classes (From Snippet 1) ---
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
        
        if check_robot_collision(new_conf, self.shapes): 
            return "Trapped", None
            
        new_node = Node(new_conf, parent=nearest)
        tree.append(new_node)
        
        if np.linalg.norm(new_conf - target_conf) < 1e-3: 
            return "Reached", new_node
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
        # Connect the two trees
        return np.array(path_a + path_b) if tree_a_is_start == 0 else np.array(path_b[::-1] + path_a[::-1])