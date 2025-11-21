#!/usr/bin/env python

"""
ROS CollisionObject Processing and Mesh Rasterization (convert geometric shapes into 2D grid obstacles)
""" 

from typing import Tuple
from .grid_2d import Grid2D, CellType


def rasterize_collision_object(collision_msg, grid: Grid2D):
    """
    Process CollisionObject message and rasterize obstacles based on object ID
    
    Args:
        collision_msg: moveit_msgs/CollisionObject message
        grid: Grid2D instance to update
    """
    try:
        if collision_msg.operation != collision_msg.ADD:
            print(f"Skipping collision object {collision_msg.id}: operation={collision_msg.operation}")
            return
        
        if not collision_msg.meshes or len(collision_msg.meshes) == 0:
            print(f"Warning: No mesh data in collision object {collision_msg.id}")
            return
        
        # Get the first mesh (Unity sends one mesh per collision object)
        mesh = collision_msg.meshes[0]
        
        if collision_msg.id == "table":
            # For table: mark workspace boundaries instead of table surface
            print("Table mesh received - marking workspace boundaries")
            rasterize_workspace_boundaries(grid, safety_margin=0.05)
            
        elif "printer" in collision_msg.id.lower():
            # For printer: rasterize as obstacle with safety margin
            print(f"Printer mesh received ({collision_msg.id}) - rasterizing obstacle")
            rasterize_mesh_as_obstacle(mesh, grid, safety_margin=0.15)
            
        else:
            # For other objects: rasterize as obstacles
            print(f"Unknown object {collision_msg.id} - rasterizing as obstacle")
            rasterize_mesh_as_obstacle(mesh, grid, safety_margin=0.10)
    
    except Exception as e:
        print(f"Error processing collision object {collision_msg.id}: {e}")


def rasterize_mesh_as_obstacle(ros_mesh, grid: Grid2D, safety_margin: float = 0.15):
    """
    Convert ROS shape_msgs/Mesh to 2D grid occupancy using bounding box approach
    
    Args:
        ros_mesh: shape_msgs/Mesh with vertices (geometry_msgs/Point[]) and triangles
        grid: Grid2D instance to update
        safety_margin: Additional clearance around mesh (meters)
    """
    try:
        if not ros_mesh.vertices or len(ros_mesh.vertices) == 0:
            print("Warning: Empty mesh vertices")
            return
        
        # Extract 2D bounding box from mesh vertices
        x_coords = [vertex.x for vertex in ros_mesh.vertices]
        y_coords = [vertex.y for vertex in ros_mesh.vertices]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        print(f"Mesh bounds: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]")
        
        # Expand bounding box by safety margin
        expanded_bounds = {
            'min_x': min_x - safety_margin,
            'max_x': max_x + safety_margin,
            'min_y': min_y - safety_margin,
            'max_y': max_y + safety_margin
        }
        
        # Mark grid cells within expanded bounds as occupied
        occupied_count = 0
        for i in range(grid.width):
            for j in range(grid.height):
                world_x, world_y = grid.grid_to_world(i, j)
                
                if (expanded_bounds['min_x'] <= world_x <= expanded_bounds['max_x'] and
                    expanded_bounds['min_y'] <= world_y <= expanded_bounds['max_y']):
                    grid.set_occupied(i, j)
                    occupied_count += 1
        
        print(f"Mesh rasterization: {occupied_count} cells marked as occupied (safety margin: {safety_margin}m)")
        
    except Exception as e:
        print(f"Error rasterizing mesh: {e}")


def rasterize_workspace_boundaries(grid: Grid2D, safety_margin: float = 0.05):
    """
    Mark workspace boundaries where robot cannot safely operate
    
    NOTE: Table surface itself is NOT an obstacle since the 2D grid operates
    ABOVE the table surface. We only mark areas where the robot arm cannot reach.
    
    Args:
        grid: Grid2D instance to update
        safety_margin: Distance from workspace edge to mark as boundary (meters)
    """
    boundary_count = 0
    
    # Mark cells near workspace boundaries as BOUNDARY
    for i in range(grid.width):
        for j in range(grid.height):
            world_x, world_y = grid.grid_to_world(i, j)
            
            # Check if near workspace boundaries
            if (world_x <= -1.5 + safety_margin or world_x >= 1.5 - safety_margin or
                world_y <= -1.5 + safety_margin or world_y >= 1.5 - safety_margin):
                grid.set_boundary(i, j)
                boundary_count += 1
    
    print(f"Workspace boundaries: {boundary_count} cells marked as boundary (margin: {safety_margin}m)")


def rasterize_rectangular_area(grid: Grid2D, bounds: Tuple[float, float, float, float]):
    """
    Helper function to mark rectangular area as occupied
    
    Args:
        grid: Grid2D instance to update
        bounds: Tuple of (min_x, min_y, max_x, max_y) in world coordinates
    """
    min_x, min_y, max_x, max_y = bounds
    occupied_count = 0
    
    for i in range(grid.width):
        for j in range(grid.height):
            world_x, world_y = grid.grid_to_world(i, j)
            
            if min_x <= world_x <= max_x and min_y <= world_y <= max_y:
                grid.set_occupied(i, j)
                occupied_count += 1
    
    print(f"Rectangular area rasterization: {occupied_count} cells marked as occupied")


def rasterize_circular_area(grid: Grid2D, center: Tuple[float, float], radius: float):
    """
    Helper function to mark circular area as occupied
    
    Args:
        grid: Grid2D instance to update
        center: (x, y) center point in world coordinates
        radius: Radius in meters
    """
    cx, cy = center
    occupied_count = 0
    
    for i in range(grid.width):
        for j in range(grid.height):
            world_x, world_y = grid.grid_to_world(i, j)
            
            # Check if point is within circular area
            distance_sq = (world_x - cx)**2 + (world_y - cy)**2
            if distance_sq <= radius**2:
                grid.set_occupied(i, j)
                occupied_count += 1
    
    print(f"Circular area rasterization: {occupied_count} cells marked as occupied (radius: {radius}m)")


def clear_collision_object(collision_msg, grid: Grid2D):
    """
    Remove collision object from grid (set cells back to FREE)
    
    Args:
        collision_msg: moveit_msgs/CollisionObject message with REMOVE operation
        grid: Grid2D instance to update
    """
    try:
        if collision_msg.operation == collision_msg.REMOVE:
            print(f"Removing collision object: {collision_msg.id}")
            
            # to do
            pass
        
    except Exception as e:
        print(f"Error removing collision object {collision_msg.id}: {e}")


def update_collision_object_pose(collision_msg, grid: Grid2D):
    """
    Update collision object pose (MOVE operation)
    
    Args:
        collision_msg: moveit_msgs/CollisionObject message with MOVE operation
        grid: Grid2D instance to update
    """
    try:
        if collision_msg.operation == collision_msg.MOVE:
            print(f"Moving collision object: {collision_msg.id}")
            
            # require tracking object positions and updating accordingly
            pass
        
    except Exception as e:
        print(f"Error moving collision object {collision_msg.id}: {e}")