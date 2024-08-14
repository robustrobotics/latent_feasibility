import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import seaborn as sns 
import pandas as pd 
from scipy.spatial.transform import Rotation as R


def load_dataset(file_path):
    with open(file_path, "rb") as handle:
        return pickle.load(handle)


def extract_final_positions_and_orientations(dataset):
    """
    Get only the final positions instead of the entire trajectories.
    """
    final_positions = []
    final_orientations = []
    for object_data in dataset:
        for push_data in object_data:
            _, (transformation, _) = push_data
            final_position = transformation[:3, 3] 
            final_orientation = R.from_matrix(transformation[:3, :3]).as_quat()
            final_positions.append(final_position)
            final_orientations.append(final_orientation)
    return np.array(final_positions), np.array(final_orientations)

def extract_contact_points(dataset): 
    contact_points = []
    for object_data in dataset: 
        for push_data in object_data: 
            attributes, (success, logs) = push_data
            contact_point = attributes[1] 
            if contact_point is None: 
                continue 
            contact_points.append(contact_point)
            # print(np.array(contact_point).shape) 


    return np.array(contact_points)


def quaternion_to_euler(q):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = q
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return np.degrees(roll_x), np.degrees(pitch_y), np.degrees(yaw_z)


def plot_final_positions_and_orientations(positions, orientations, title, output_dir):
    """
    Plot the final positions as a way of ensuring the dataset's distribution is good.
    There are multiple different plots that will be generated and saved in the directory specified in args.
    """
    # Convert quaternions to Euler angles in degrees

    far = 0
    total = 0
    for x, y, z in positions: 
        if x > 1.0 or x < -1.0 or y > 1.0 or y < -1.0 or z > 1.0 or z < -1.0:
            far += 1 
        total += 1
    print(f"{far}/{total} are far away")

    distances = np.linalg.norm(positions[:,:2], axis=1)
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame(distances, columns=["Distance (m)"]) 
    sns.displot(df, x="Distance (m)")
 
    plot_path = os.path.join(output_dir, "distance_plot.png")
    plt.savefig(plot_path)
    plt.close()

    euler_angles = np.array([quaternion_to_euler(q) for q in orientations])
    roll_degrees, pitch_degrees, yaw_degrees = (
        euler_angles[:, 0],
        euler_angles[:, 1],
        euler_angles[:, 2],
    )

    # Plot positions and orientations
    fig = plt.figure(figsize=(20, 25))

    # 2D position plot
    ax1 = fig.add_subplot(331)
    scatter = ax1.scatter(
        positions[:, 0], positions[:, 1], c=yaw_degrees, cmap="hsv", alpha=0.5
    )
    ax1.set_title(f"{title}: Final Positions")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.axis("equal")
    ax1.grid(True)
    plt.colorbar(scatter, ax=ax1, label="Yaw Angle (degrees)")

    # 3D position and orientation plot
    ax2 = fig.add_subplot(332, projection="3d")
    scatter3d = ax2.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=yaw_degrees,
        cmap="hsv",
        alpha=0.5,
    )
    ax2.set_title(f"{title}: 3D Positions and Orientations")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.set_zlabel("Z Position")
    plt.colorbar(scatter3d, ax=ax2, label="Yaw Angle (degrees)")

    # Roll angle distribution plot
    ax3 = fig.add_subplot(334)
    ax3.hist(roll_degrees, bins=36, range=(-180, 180), edgecolor="black")
    ax3.set_title(f"{title}: Distribution of Roll Angles")
    ax3.set_xlabel("Roll Angle (degrees)")
    ax3.set_ylabel("Frequency")
    ax3.set_xlim(-180, 180)
    ax3.grid(True)

    # Pitch angle distribution plot
    ax4 = fig.add_subplot(335)
    ax4.hist(pitch_degrees, bins=36, range=(-180, 180), edgecolor="black")
    ax4.set_title(f"{title}: Distribution of Pitch Angles")
    ax4.set_xlabel("Pitch Angle (degrees)")
    ax4.set_ylabel("Frequency")
    ax4.set_xlim(-180, 180)
    ax4.grid(True)

    # Yaw angle distribution plot
    ax5 = fig.add_subplot(336)
    ax5.hist(yaw_degrees, bins=36, range=(-180, 180), edgecolor="black")
    ax5.set_title(f"{title}: Distribution of Yaw Angles")
    ax5.set_xlabel("Yaw Angle (degrees)")
    ax5.set_ylabel("Frequency")
    ax5.set_xlim(-180, 180)
    ax5.grid(True)

    # Polar plot of roll angles
    ax6 = fig.add_subplot(337, projection="polar")
    ax6.hist(np.radians(roll_degrees), bins=36, range=(-np.pi, np.pi))
    ax6.set_title(f"{title}: Polar Distribution of Roll Angles")
    ax6.set_theta_zero_location("N")
    ax6.set_theta_direction(-1)
    ax6.set_thetagrids(np.arange(0, 360, 30))
    ax6.set_ylim(0, ax6.get_ylim()[1])

    # Polar plot of pitch angles
    ax7 = fig.add_subplot(338, projection="polar")
    ax7.hist(np.radians(pitch_degrees), bins=36, range=(-np.pi, np.pi))
    ax7.set_title(f"{title}: Polar Distribution of Pitch Angles")
    ax7.set_theta_zero_location("N")
    ax7.set_theta_direction(-1)
    ax7.set_thetagrids(np.arange(0, 360, 30))
    ax7.set_ylim(0, ax7.get_ylim()[1])

    # Polar plot of yaw angles
    ax8 = fig.add_subplot(339, projection="polar")
    ax8.hist(np.radians(yaw_degrees), bins=36, range=(-np.pi, np.pi))
    ax8.set_title(f"{title}: Polar Distribution of Yaw Angles")
    ax8.set_theta_zero_location("N")
    ax8.set_theta_direction(-1)
    ax8.set_thetagrids(np.arange(0, 360, 30))
    ax8.set_ylim(0, ax8.get_ylim()[1])

    # Heatmap of 2D positions
    ax9 = fig.add_subplot(333)
    heatmap, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    cax = ax9.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
    ax9.set_title(f"{title}: 2D Position Heatmap")
    ax9.set_xlabel("X Position (m)")
    ax9.set_ylabel("Y Position (m)")
    fig.colorbar(cax, ax=ax9, label="Frequency")
    ax10 = fig.add_subplot(3, 4, 10)
    ax10.hist(positions[:, 2], bins=50, edgecolor='black')
    ax10.set_title(f"{title}: Distribution of Z Values")
    ax10.set_xlabel("Z Position (m)")
    ax10.set_ylabel("Frequency")
    ax10.grid(True)
    # Adjust layout and save the plot
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_contact_points(points, output_dir):
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(111)
    scatter = ax1.scatter(
        points[:, 0], points[:, 1], alpha=0.5
    )
    ax1.set_title(f"Contact Points")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.axis("equal")
    ax1.grid(True)

    # Adjust layout and save the plot
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"contact_points.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")

def main(args):
    data_root_path = os.path.join("learning", "data", "pushing", args.dataset_name)

    # Create output directory for plots
    output_dir = os.path.join(data_root_path, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Load and plot training data
    train_path = os.path.join(data_root_path, "train_dataset.pkl")


    train_data = load_dataset(train_path)
    train_contact_points = extract_contact_points(train_data)
    # plot_contact_points(train_contact_points, output_dir)    

    train_positions, train_orientations = extract_final_positions_and_orientations(
        train_data
    )
    plot_final_positions_and_orientations(
        train_positions, train_orientations, "Training Data", output_dir
    )

    # Load and plot test data
    test_path = os.path.join(data_root_path, "test_dataset.pkl")
    test_data = load_dataset(test_path)
    test_positions, test_orientations = extract_final_positions_and_orientations(
        test_data
    )
    plot_final_positions_and_orientations(
        test_positions, test_orientations, "Test Data", output_dir
    )

    # Load and plot same geometry test data
    samegeo_test_path = os.path.join(data_root_path, "samegeo_test_dataset.pkl")
    samegeo_test_data = load_dataset(samegeo_test_path)
    samegeo_test_positions, samegeo_test_orientations = (
        extract_final_positions_and_orientations(samegeo_test_data)
    )
    plot_final_positions_and_orientations(
        samegeo_test_positions,
        samegeo_test_orientations,
        "Same Geometry Test Data",
        output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize final positions and orientations of sliders from pushing data."
    )
    parser.add_argument(
        "dataset_name", type=str, help="Name of the dataset root directory"
    )
    args = parser.parse_args()
    main(args)
