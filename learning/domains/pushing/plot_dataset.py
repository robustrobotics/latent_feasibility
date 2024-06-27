import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from mpl_toolkits.mplot3d import Axes3D

def load_dataset(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def extract_final_positions_and_orientations(dataset):
    final_positions = []
    final_orientations = []
    for object_data in dataset:
        for push_data in object_data:
            _, (success, logs) = push_data
            if logs:  # Check if logs exist
                final_position = logs[-1][:3]  # Get the last logged position
                final_orientation = logs[-1][3:]  # Get the last logged orientation
                final_positions.append(final_position)
                final_orientations.append(final_orientation)
    return np.array(final_positions), np.array(final_orientations)

def quaternion_to_euler(q):
    # Convert quaternion to Euler angles
    w, x, y, z = q
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def plot_final_positions_and_orientations(positions, orientations, title, output_dir):
    # Convert quaternions to Euler angles in degrees
    euler_angles = np.array([quaternion_to_euler(q) for q in orientations])
    yaw_degrees = euler_angles[:, 2]  # Extract yaw (rotation about z-axis)

    # Plot positions and orientations
    fig = plt.figure(figsize=(20, 15))
    
    # 2D position plot
    ax1 = fig.add_subplot(221)
    scatter = ax1.scatter(positions[:, 0], positions[:, 1], c=yaw_degrees, cmap='hsv', alpha=0.5)
    ax1.set_title(f'{title}: Final Positions')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.axis('equal')
    ax1.grid(True)
    plt.colorbar(scatter, ax=ax1, label='Yaw Angle (degrees)')

    # 3D position and orientation plot
    ax2 = fig.add_subplot(222, projection='3d')
    scatter3d = ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=yaw_degrees, cmap='hsv', alpha=0.5)
    ax2.set_title(f'{title}: 3D Positions and Orientations')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_zlabel('Z Position')
    plt.colorbar(scatter3d, ax=ax2, label='Yaw Angle (degrees)')

    # Yaw angle distribution plot
    ax3 = fig.add_subplot(223)
    ax3.hist(yaw_degrees, bins=36, range=(-180, 180), edgecolor='black')
    ax3.set_title(f'{title}: Distribution of Yaw Angles')
    ax3.set_xlabel('Yaw Angle (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.set_xlim(-180, 180)
    ax3.grid(True)

    # Polar plot of yaw angles
    ax4 = fig.add_subplot(224, projection='polar')
    ax4.hist(np.radians(yaw_degrees), bins=36, range=(-np.pi, np.pi))
    ax4.set_title(f'{title}: Polar Distribution of Yaw Angles')
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)
    ax4.set_thetagrids(np.arange(0, 360, 30))
    ax4.set_ylim(0, ax4.get_ylim()[1])  # Set radial limit to start from 0

    # Adjust layout and save the plot
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def main(args):
    data_root_path = os.path.join('learning', 'data', 'pushing', args.dataset_name)
    
    # Create output directory for plots
    output_dir = os.path.join(data_root_path, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and plot training data
    train_path = os.path.join(data_root_path, 'train_dataset.pkl')
    train_data = load_dataset(train_path)
    train_positions, train_orientations = extract_final_positions_and_orientations(train_data)
    plot_final_positions_and_orientations(train_positions, train_orientations, 'Training Data', output_dir)

    # Load and plot test data
    test_path = os.path.join(data_root_path, 'test_dataset.pkl')
    test_data = load_dataset(test_path)
    test_positions, test_orientations = extract_final_positions_and_orientations(test_data)
    plot_final_positions_and_orientations(test_positions, test_orientations, 'Test Data', output_dir)

    # Load and plot same geometry test data
    samegeo_test_path = os.path.join(data_root_path, 'samegeo_test_dataset.pkl')
    samegeo_test_data = load_dataset(samegeo_test_path)
    samegeo_test_positions, samegeo_test_orientations = extract_final_positions_and_orientations(samegeo_test_data)
    plot_final_positions_and_orientations(samegeo_test_positions, samegeo_test_orientations, 'Same Geometry Test Data', output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize final positions and orientations of sliders from pushing data.")
    parser.add_argument('dataset_name', type=str, help="Name of the dataset root directory")
    args = parser.parse_args()
    main(args)