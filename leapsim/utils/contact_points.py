import pybullet as p
import pybullet_data
import time
import numpy as np
import transforms3d as tf3d
import os
import trimesh
import logging
from typing import List, Dict, Tuple, Optional, Union
from sklearn.neighbors import KDTree

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_npy_data(prefix: str = "") -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Loads the npy data files"""
    try:
        object_poses = np.load(prefix + "object_pose_history.npy", allow_pickle=True)
        hand_base_poses = np.load(prefix + "hand_base_pose_history.npy", allow_pickle=True)
        hand_joint_poses = np.load(prefix + "hand_joint_pose_history.npy")

        # Validate shapes
        if not isinstance(object_poses, np.ndarray) or object_poses.ndim != 1:
            logging.error(f"Invalid shape for object_poses: {object_poses.shape if hasattr(object_poses, 'shape') else 'Not an ndarray'}")
            return None, None, None
        if not isinstance(hand_base_poses, np.ndarray) or hand_base_poses.ndim != 1:
            logging.error(f"Invalid shape for hand_base_poses: {hand_base_poses.shape if hasattr(hand_base_poses, 'shape') else 'Not an ndarray'}")
            return None, None, None
        if not isinstance(hand_joint_poses, np.ndarray) or hand_joint_poses.ndim != 2:
            logging.error(f"Invalid shape for hand_joint_poses: {hand_joint_poses.shape if hasattr(hand_joint_poses, 'shape') else 'Not an ndarray'}")
            return None, None, None
        
        logging.info(f"Loaded data from {prefix}")
        return object_poses, hand_base_poses, hand_joint_poses
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Make sure the .npy files are in the correct path")
        return None, None, None

def euler_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """Converts a quaternion to euler angles (rpy)"""
    # Convert quaternion from [x, y, z, w] to [w, x, y, z]
    quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]
    rpy = tf3d.euler.quat2euler(quat_wxyz)
    return rpy

def unscale_np(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return (2.0 * x - upper - lower)/(upper - lower)

def load_meshes_from_body(body_id: int, use_convex_hull: bool = True) -> List[trimesh.Trimesh]:
    """Loads all meshes (visual shapes) associated with a PyBullet body ID and optionally simplifies them using a convex hull."""
    meshes = []
    for link_index in range(-1, p.getNumJoints(body_id)):
        visual_shapes = p.getVisualShapeData(body_id, link_index)
        for shape_data in visual_shapes:
            if shape_data[2] == p.GEOM_MESH:
                mesh_path = shape_data[4].decode("utf-8")
                try:
                    mesh = trimesh.load_mesh(mesh_path)
                    if use_convex_hull:
                         if len(mesh.vertices) > 0:
                             mesh = mesh.convex_hull
                         else:
                              logging.warning(f"Warning: Mesh has no vertices at {mesh_path}. Using the original mesh")
                    meshes.append(mesh)
                except Exception as e:
                    logging.error(f"Error loading mesh at {mesh_path}: {e}")
                    continue
            elif shape_data[2] == p.GEOM_BOX:
                box_half_extents = np.array(shape_data[3])/2.0
                mesh = trimesh.creation.box(extents=box_half_extents*2)
                if use_convex_hull:
                    mesh = mesh.convex_hull
                meshes.append(mesh)
    return meshes

def transform_meshes(body_id: int, meshes: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
    """Transforms a list of trimesh meshes based on the current pose of a PyBullet body."""
    transformed_meshes = []
    for link_index in range(-1, p.getNumJoints(body_id)): # loop through the links of the body ID
      
        mesh = meshes[link_index+1] if link_index > -1 else meshes[0] # since the mesh has been loaded with base mesh at index 0 and link meshes from 1 on wards
        
        if link_index == -1: # base link of the mesh
            base_pos, base_ori = p.getBasePositionAndOrientation(body_id)
            transform = np.eye(4)
            transform[:3,:3] = tf3d.quaternions.quat2mat(base_ori)
            transform[:3,3] = base_pos
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_transform(transform)
            transformed_meshes.append(transformed_mesh)
        else: # for the links of the mesh
             link_state = p.getLinkState(body_id, link_index)
             if link_state is None:
                 logging.warning(f"Warning: Link state is None for index {link_index}. Skipping this link")
                 continue
             link_pos, link_ori = link_state[0], link_state[1]
             transform = np.eye(4)
             transform[:3,:3] = tf3d.quaternions.quat2mat(link_ori)
             transform[:3,3] = link_pos
             transformed_mesh = mesh.copy()
             transformed_mesh.apply_transform(transform)
             transformed_meshes.append(transformed_mesh)
    return transformed_meshes

def check_mesh_intersections(mesh_hand_list, mesh_obj_list, threshold=0.2, num_points=50000):
    """
    Optimized mesh intersection checking using KD-Trees.

    Args:
        mesh_hand_list: List of hand mesh trimesh objects
        mesh_obj_list: List of object mesh trimesh objects
        threshold: Distance threshold for considering points as intersecting (meters)
    Returns:
        numpy.ndarray: Array of intersection points
    """
    intersection_points = []

    if not mesh_obj_list:
        return np.empty((0, 3))
    
    for mesh_obj in mesh_obj_list:
        # Build KDTree for the object mesh (do this once for each object mesh)
        if len(mesh_obj.vertices) > 0:
            obj_kdtree = KDTree(mesh_obj.vertices)
        else:
            logging.warning(f"Warning: Object mesh has no vertices. Skipping intersection check for object mesh")
            continue
        for mesh_hand in mesh_hand_list:
            
            # Generate point clouds on the hand mesh
            points_hand, _ = mesh_hand.sample(num_points, return_index=True)
            if not points_hand.any():
               logging.debug(f"Skipping mesh, empty point cloud for hand mesh: {mesh_hand}")
               continue
           
            # Use the KDTree to find nearest neighbors
            distances, _ = obj_kdtree.query(points_hand, k=1)
            intersecting_points = points_hand[distances.flatten() < threshold]
            
            if len(intersecting_points) > 0:
                intersection_points.extend(intersecting_points)
                logging.debug(f"Found {len(intersecting_points)} intersection points")
            
            if not intersection_points and distances.flatten().min() > threshold: # early exit if no intersection
                  continue
            

    return np.array(intersection_points) if intersection_points else np.empty((0, 3))


def visualize_scene(object_urdf_path: str, hand_urdf_path: str, prefix: str = "", save_path: str ="./", time_scale: float=1.0, use_convex_hull:bool=True) -> None:
    """
    Visualizes the scene, detects and saves intersection points.
    """

    object_poses, hand_base_poses, hand_joint_poses = load_npy_data(prefix)

    if object_poses is None or hand_base_poses is None or hand_joint_poses is None:
        return

    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0) # disable real time simulation

    # Load floor
    planeId = p.loadURDF("plane.urdf")

    # Load object and hand URDFs
    object_id = p.loadURDF(object_urdf_path, basePosition=[0, 0, 0])
    hand_id = p.loadURDF(hand_urdf_path, basePosition=[0, 0, 0], flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    if hand_id < 0:
       logging.error("ERROR: URDF cannot be loaded.")
       p.disconnect()
       return
    else:
       logging.info("URDF loaded successfully")

    # Define the sim_to_real_indices
    sim_to_real_indices = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]


    num_time_steps = len(object_poses)
    
    # Get the joint index of the hand
    num_joints = p.getNumJoints(hand_id)
    joint_indices = list(range(num_joints))
    
    logging.info(f"Number of joints: {num_joints}")

    if num_joints != len(sim_to_real_indices):
        logging.error("ERROR: Number of joints in URDF does not match number of mapping indices")
        p.disconnect()
        return
    
    # Disable collisions between hand and object
    for j in range(p.getNumJoints(hand_id)):
        p.setCollisionFilterGroupMask(hand_id, j, 0, 0) # set to group 0 and mask 0, disables all collisions for the hand links
    for j in range(p.getNumJoints(object_id)):
        p.setCollisionFilterGroupMask(object_id, j, 0, 0) # set to group 0 and mask 0, disables all collisions for the object links
    
    # Set the initial camera position and orientation
    camera_distance = 0.2
    camera_yaw = 60
    camera_pitch = -10
    camera_target_position = [0, 0, 0.5]  # center of the object scene 

    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

    joint_names_to_indices = {}
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(hand_id, joint_index)
        joint_name = joint_info[1].decode('utf-8')  # Get the joint name and decode it
        joint_names_to_indices[joint_name] = joint_index
        # print(f"Joint Index: {joint_index}, Joint Name: {joint_name}")

    # Get the mesh data for the hand and object
    meshes_hand = load_meshes_from_body(hand_id, use_convex_hull)
    meshes_object = load_meshes_from_body(object_id, use_convex_hull)
    logging.info(f"Found {len(meshes_hand)} meshes for hand")
    logging.info(f"Found {len(meshes_object)} meshes for object")

    all_intersection_points = [] # list to store the intersection points

    try:
        for i in range(num_time_steps):
            logging.info(f"Time step: {i}")
            # Set Object pose
            obj_pos = object_poses[i]['position']
            obj_quat = object_poses[i]['orientation']
            obj_rpy = euler_from_quaternion(obj_quat)
            p.resetBasePositionAndOrientation(object_id, obj_pos, p.getQuaternionFromEuler(obj_rpy))

            # Set Hand Base pose
            hand_pos = hand_base_poses[i]['position']
            hand_quat = hand_base_poses[i]['orientation']
            hand_rpy = euler_from_quaternion(hand_quat)
            p.resetBasePositionAndOrientation(hand_id, hand_pos, p.getQuaternionFromEuler(hand_rpy))
            
            # Remap the hand joint positions using sim_to_real_indices
            hand_joint_positions = hand_joint_poses[i]
            remapped_joint_positions = sim_to_real(hand_joint_positions, sim_to_real_indices)
            
            # Ensure the number of joints matches the length of joint positions data
            if len(remapped_joint_positions) != num_joints:
                logging.warning(f"Warning: Number of joints {num_joints} in URDF does not match number of joint positions: {len(remapped_joint_positions)} in step {i}. Skipping the current time step. Please verify your URDF and data file")
                continue

            for joint_index, joint_pos in zip(joint_indices, remapped_joint_positions):
                p.resetJointState(hand_id, joint_index, joint_pos)
            logging.debug(f"Set hand joints at step {i}: {remapped_joint_positions}")
            # Get the transformed meshes
            transformed_meshes_hand = transform_meshes(hand_id, meshes_hand)
            transformed_meshes_object = transform_meshes(object_id, meshes_object)


            intersection_points = check_mesh_intersections(transformed_meshes_hand, transformed_meshes_object, threshold=0.005)

            # extend the global list of intersection points
            all_intersection_points.extend(intersection_points)
            
            # Visualize intersection points
            if len(intersection_points) > 0:
                for point in intersection_points:
                     p.addUserDebugLine(point, point + np.array([0,0,0.001]), [1, 0, 0], lifeTime=0.1, lineWidth=8)
        
            p.stepSimulation()
            time.sleep(0.02/time_scale) # control the speed of simulation

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        p.disconnect()

        # Save the intersection points to a NumPy array
        if all_intersection_points:
            intersection_points_array = np.array(all_intersection_points)
            
            # ensure the save directory exists
            os.makedirs(save_path, exist_ok=True)
            
            file_path = os.path.join(save_path,"intersection_points.npy")

            np.save(file_path, intersection_points_array)
            logging.info(f"Intersection points saved to: {file_path}")
        else:
           logging.info("No intersection points to save")


def sim_to_real(values: np.ndarray, sim_to_real_indices: List[int]) -> np.ndarray:
    """
    Applies the sim to real index transformation using numpy
    """
    return values[sim_to_real_indices]

if __name__ == '__main__':
    # Paths to your URDF files and npy data
    hand_urdf_path = "/home/control-lab/LEAP_Hand_Sim/assets/leap_hand/robot.urdf"  # Path to your object URDF
    object_urdf_path = "/home/control-lab/LEAP_Hand_Sim/assets/cube.urdf"   # Path to your hand URDF
    npy_file_prefix = "/home/control-lab/LEAP_Hand_Sim/leapsim/" # Prefix for the .npy data files. can be a path if needed
    save_path = "/home/control-lab/LEAP_Hand_Sim/leapsim_results/" # Path to save the intersection points
    # Adjust the speed of simulation
    time_scale = 1 # slow down the simulation
    use_convex_hull = True # Set this to True to enable the usage of convex hulls

    visualize_scene(object_urdf_path, hand_urdf_path, npy_file_prefix, save_path, time_scale, use_convex_hull)