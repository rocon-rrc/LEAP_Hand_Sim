import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import json
import argparse

def analyze_simulation_data(directory):
    """
    Analyzes the simulation data stored in .npy and .json files.
    Args:
        directory (str): Directory where the data files are located.
    """

    # Load all npy files
    object_pose_history = np.load(os.path.join(directory, "object_pose_history.npy"), allow_pickle=True)
    hand_base_pose_history = np.load(os.path.join(directory, "hand_base_pose_history.npy"), allow_pickle=True)
    hand_joint_pose_history = np.load(os.path.join(directory, "hand_joint_pose_history.npy"), allow_pickle=True)
    image_history = np.load(os.path.join(directory, "image_history.npy"), allow_pickle=True)
    depth_history = np.load(os.path.join(directory, "depth_history.npy"), allow_pickle=True)
    contact_history = np.load(os.path.join(directory, "contact_history.npy"), allow_pickle=True)

    # Load json files
    with open(os.path.join(directory, "camera_intrinsics.json"), "r") as f:
        camera_intrinsics = json.load(f)
    with open(os.path.join(directory, "camera_view.json"), "r") as f:
        camera_view = json.load(f)
    with open(os.path.join(directory, "rigid_body_indices.json"), "r") as f:
        rigid_body_indices = json.load(f)

    # Print shapes
    print("----- Data Shapes -----")
    print(f"Object Pose History: {object_pose_history.shape}")
    print(f"Hand Base Pose History: {hand_base_pose_history.shape}")
    print(f"Hand Joint Pose History: {hand_joint_pose_history.shape}")
    print(f"Image History: {image_history.shape}")
    print(f"Depth History: {depth_history.shape}")
    print(f"Contact History: {contact_history.shape}")
    print("-----------------------")

    # Print camera intrinsics, view matrix and rigid body indices
    print("----- Camera and Rigid Body Data -----")
    print(f"Camera Intrinsics: {camera_intrinsics}")
    print(f"Camera View Matrix: {np.array(camera_view)}")
    print(f"Rigid Body Indices: {rigid_body_indices}")
    print("--------------------------------------")

    # Display video if image data is present
    if len(image_history.shape) > 1 and image_history.shape[0] > 1:
        fig, ax = plt.subplots()
        im = ax.imshow(image_history[0])

        def update(frame):
            im.set_array(image_history[frame])
            return im,

        ani = animation.FuncAnimation(fig, update, frames=len(image_history), interval=50, blit=True)

        plt.show()
        
    if len(depth_history.shape) > 1 and depth_history.shape[0] > 1:
        fig, ax = plt.subplots()
        im = ax.imshow(depth_history[0])
        def update(frame):
           im.set_array(depth_history[frame])
           return im,
        ani = animation.FuncAnimation(fig, update, frames=len(depth_history), interval=50, blit=True)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze simulation data.")
    parser.add_argument("directory", type=str, help="Directory containing the simulation data.")
    args = parser.parse_args()
    
    analyze_simulation_data(args.directory)