import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def analyze_simulation_data(directory="."):
    """
    Analyzes the simulation data stored in .npy files.
    Args:
        directory (str, optional): Directory where the .npy files are located. Defaults to ".".
    """

    # Load all npy files
    object_pose_history = np.load(os.path.join(directory, "object_pose_history.npy"), allow_pickle=True)
    hand_base_pose_history = np.load(os.path.join(directory, "hand_base_pose_history.npy"), allow_pickle=True)
    hand_joint_pose_history = np.load(os.path.join(directory, "hand_joint_pose_history.npy"), allow_pickle=True)
    image_history = np.load(os.path.join(directory, "image_history.npy"), allow_pickle=True)
    depth_history = np.load(os.path.join(directory, "depth_history.npy"), allow_pickle=True)
    contact_history = np.load(os.path.join(directory, "contact_history.npy"), allow_pickle=True)


    # Print shapes
    print("----- Data Shapes -----")
    print(f"Object Pose History: {object_pose_history.shape}")
    print(f"Hand Base Pose History: {hand_base_pose_history.shape}")
    print(f"Hand Joint Pose History: {hand_joint_pose_history.shape}")
    print(f"Image History: {image_history.shape}")
    print(f"Depth History: {depth_history.shape}")
    print(f"Contact History: {contact_history.shape}")
    print("-----------------------")


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
    analyze_simulation_data("/home/control-lab/LEAP_Hand_Sim/leapsim")