#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# No longer using gridspec
from learning.domains.pushing.virtual_tables import BoxTable, RingTable, BeamTable

# Define hardcoded view centers and boundaries for each table type
# This ensures consistent visualization regardless of table implementation details
BOX_VIEW_CENTER = [2.0, 2.0]      # Center of box table
BOX_VIEW_SIZE = 4.0               # Size of view window
RING_VIEW_CENTER = [2.5, 2.5]     # Center of ring table
RING_VIEW_SIZE = 5.0              # Size of view window
BEAM_VIEW_CENTER = [2.0, 1.0]     # Offset center of beam table to show both platforms
BEAM_VIEW_SIZE = 5.0              # Larger size to ensure goal is visible

def visualize_table(ax, table, title, block_pos=None, goal_pos=None, res=0.01, show_colorbar=False, margins=0.2, center_view=False):
    """
    Visualize a SimulatedTable on a matplotlib axis.
    
    Args:
        ax: Matplotlib axis to plot on
        table: SimulatedTable instance
        title: Title for the plot
        block_pos: Optional position for a block to be shown on the table
        goal_pos: Optional position for a goal to be shown
        res: Resolution for visualization
        show_colorbar: Whether to show the colorbar
    """
    # Get table dimensions if available, otherwise estimate
    if hasattr(table, 'table_length') and hasattr(table, 'table_width'):
        min_x, max_x = 0, table.table_length
        min_y, max_y = 0, table.table_width
    elif hasattr(table, 'outer_rad'):
        # For RingTable, estimate dimensions from the outer radius and center
        ring_center = getattr(table, 'ring_center', np.array([0, 0]))
        outer_rad = getattr(table, 'outer_rad', 1.0)
        min_x, max_x = ring_center[0] - outer_rad*1.5, ring_center[0] + outer_rad*1.5
        min_y, max_y = ring_center[1] - outer_rad*1.5, ring_center[1] + outer_rad*1.5
    else:
        # Default values
        min_x, max_x = -2, 2
        min_y, max_y = -2, 2
        
    # Ensure start and goal positions are in the frame
    if hasattr(table, 'start_pos'):
        min_x = min(min_x, table.start_pos[0] - margins)
        max_x = max(max_x, table.start_pos[0] + margins)
        min_y = min(min_y, table.start_pos[1] - margins)
        max_y = max(max_y, table.start_pos[1] + margins)
    
    if hasattr(table, 'goal_pos'):
        min_x = min(min_x, table.goal_pos[0] - margins)
        max_x = max(max_x, table.goal_pos[0] + margins)
        min_y = min(min_y, table.goal_pos[1] - margins)
        max_y = max(max_y, table.goal_pos[1] + margins)
        
    # For any table type, ensure it's properly centered in the view
    if center_view:
        # Override the calculated boundaries with hardcoded view windows
        # based on the table type for perfect centering
        if isinstance(table, RingTable):
            # For RingTable, use the ring center-based view
            center_point = np.array(RING_VIEW_CENTER)
            half_size = RING_VIEW_SIZE / 2
            # Set exact boundaries for perfect centering
            min_x = center_point[0] - half_size
            max_x = center_point[0] + half_size
            min_y = center_point[1] - half_size
            max_y = center_point[1] + half_size
        elif isinstance(table, BeamTable):
            # For BeamTable, use the beam center-based view
            center_point = np.array(BEAM_VIEW_CENTER)
            half_size = BEAM_VIEW_SIZE / 2
            # Set exact boundaries for perfect centering
            min_x = center_point[0] - half_size
            max_x = center_point[0] + half_size
            min_y = center_point[1] - half_size
            max_y = center_point[1] + half_size
    
    # Create grid
    x = np.arange(min_x, max_x, res)
    y = np.arange(min_y, max_y, res)
    X, Y = np.meshgrid(x, y)
    
    # Compute table height field
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = table.is_on_table_fn(np.array([X[i, j], Y[i, j]]))
    
    # Fill the table surface with a light blue color
    binary_mask = (Z >= 0).astype(float)
    ax.imshow(binary_mask, extent=[min_x, max_x, min_y, max_y], 
              origin='lower', cmap=plt.cm.Blues, alpha=0.3, vmin=0, vmax=1)
    
    # Draw the boundary by finding the contour between positive and negative values
    # We'll do this manually to avoid matplotlib warnings
    from scipy import ndimage
    mask = Z >= 0
    # Apply edge detection to find the boundary
    edge = ndimage.binary_erosion(mask) ^ mask
    # Plot just the edge pixels
    edge_y, edge_x = np.where(edge)
    if len(edge_x) > 0 and len(edge_y) > 0:
        # Convert indices back to coordinate space
        edge_x = min_x + edge_x * (max_x - min_x) / (X.shape[1] - 1)
        edge_y = min_y + edge_y * (max_y - min_y) / (X.shape[0] - 1)
        ax.scatter(edge_x, edge_y, s=0.5, color='#2F5D8C', alpha=1)
    

    
    # Set title and remove ticks
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot start and goal positions
    if hasattr(table, 'start_pos'):
        ax.plot(table.start_pos[0], table.start_pos[1], 'o', 
                color='green', markersize=10, label='Start')
    
    if hasattr(table, 'goal_pos'):
        ax.plot(table.goal_pos[0], table.goal_pos[1], '*', 
                color='red', markersize=12, label='Goal')
    
    # Add a block if position is provided, or use the initial pose from the table
    if block_pos is None and hasattr(table, 'init_pose'):
        # Extract position and orientation from initial pose matrix
        init_x = table.init_pose[0, 2]
        init_y = table.init_pose[1, 2]
        # Extract orientation (arctan2 of the rotation matrix elements)
        init_angle = np.arctan2(table.init_pose[1, 0], table.init_pose[0, 0])
        block_pos = np.array([init_x, init_y, init_angle])
    
    if block_pos is not None:
        if len(block_pos) >= 2:
            angle = block_pos[2] if len(block_pos) > 2 else 0
            block_width = table.block_width
            block_length = table.block_length
            
            # Create a rotated rectangle for the block
            block_x, block_y = block_pos[0], block_pos[1]
            dx = block_length / 2
            dy = block_width / 2
            
            corners = np.array([
                [-dx, -dy],
                [dx, -dy],
                [dx, dy],
                [-dx, dy]
            ])
            
            # Rotate and translate corners
            rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            rotated_corners = np.array([rot_matrix.dot(corner) for corner in corners])
            translated_corners = rotated_corners + np.array([block_x, block_y])
            
            # Plot the block
            block = Polygon(translated_corners, closed=True, 
                          facecolor='gray', alpha=0.8, edgecolor='black')
            ax.add_patch(block)
            
            # Plot center as a small red point (for simplicity we'll just show the center)
            # Instead of trying to access block_com_relative_to_centroid which isn't stored directly
            ax.plot(block_x, block_y, 'ro', markersize=5)
    
    return ax

def create_table_figure(save_path=None):
    """Create a figure with all three table types for a publication."""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Use equal spacing for subplots with more width between them
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.85, wspace=0.4)
    
    # Create the subplots with the proper aspect ratio
    ax1 = fig.add_subplot(131, aspect='equal')
    ax2 = fig.add_subplot(132, aspect='equal')
    ax3 = fig.add_subplot(133, aspect='equal')
    
    # Using global view parameters defined at module level
    
    # Create tables with standard parameters
    box_table = BoxTable(
        block_width=0.1,
        block_length=0.2,
        block_com_relative_to_centroid=np.array([0, 0]),
        goal_loc_x=3.0,
        goal_loc_y=3.0,
        table_length=4.0,
        table_width=4.0
    )
    
    # Use default parameters for RingTable to get default start points
    ring_table = RingTable(
        block_width=0.1,
        block_length=0.2,
        block_com_relative_to_centroid=np.array([0, 0]),
        ring_center=np.ones(2) * 2.5,  # Use default from class
        outer_rad=2.0,
        inner_rad=1.25,
        platform_rad=0.5
    )
    
    # Use default parameters for BeamTable to get default start points
    beam_table = BeamTable(
        block_width=0.1,
        block_length=0.2,
        block_com_relative_to_centroid=np.array([0, 0]),
        table_length=3.0,  # Shorter to ensure visibility
        narrow_width=0.6,
        platform_rad=0.7
    )
    
    # Use the tables' start positions for block positions
    box_block_pos = np.array([1, 1, 0])  # Custom position for box table
    # ring_block_pos = ring_table.start_pos.copy()
    # beam_block_pos = beam_table.start_pos.copy()
    
    # Subplots already created above with equal aspect ratio
    
    # Visualize tables with hardcoded boundaries for consistent display
    # Use the box_view parameters for box table to ensure consistent display
    ax1.set_xlim(BOX_VIEW_CENTER[0] - BOX_VIEW_SIZE/2, BOX_VIEW_CENTER[0] + BOX_VIEW_SIZE/2)
    ax1.set_ylim(BOX_VIEW_CENTER[1] - BOX_VIEW_SIZE/2, BOX_VIEW_CENTER[1] + BOX_VIEW_SIZE/2)
    visualize_table(ax1, box_table, "Box Table", block_pos=box_block_pos, margins=0.5)
    
    # Use hardcoded view boundaries
    ax2.set_xlim(RING_VIEW_CENTER[0] - RING_VIEW_SIZE/2, RING_VIEW_CENTER[0] + RING_VIEW_SIZE/2)
    ax2.set_ylim(RING_VIEW_CENTER[1] - RING_VIEW_SIZE/2, RING_VIEW_CENTER[1] + RING_VIEW_SIZE/2)
    visualize_table(ax2, ring_table, "Ring Table", block_pos=None, margins=0.8, center_view=True)
    
    ax3.set_xlim(BEAM_VIEW_CENTER[0] - BEAM_VIEW_SIZE/2, BEAM_VIEW_CENTER[0] + BEAM_VIEW_SIZE/2)
    ax3.set_ylim(BEAM_VIEW_CENTER[1] - BEAM_VIEW_SIZE/2, BEAM_VIEW_CENTER[1] + BEAM_VIEW_SIZE/2)
    visualize_table(ax3, beam_table, "Beam Table", block_pos=None, margins=0.8, center_view=True)
    
    # Add a main title
    plt.suptitle("Table Environments for Push Planning", fontsize=16, y=0.98)
    
    # Give the figure a bit more space at the top for the title
    fig.subplots_adjust(top=0.85)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Figure saved to %s" % save_path)
    else:
        plt.show()
    
    return fig

if __name__ == "__main__":
    # Set the output directory to the paper_figures directory
    output_dir = os.path.join('learning', 'data', 'pushing', 'paper_figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    save_path = os.path.join(output_dir, 'table_environments.png')
    create_table_figure(save_path)
    
    # Also save a PDF version for papers
    save_path_pdf = os.path.join(output_dir, 'table_environments.pdf')
    create_table_figure(save_path_pdf)
