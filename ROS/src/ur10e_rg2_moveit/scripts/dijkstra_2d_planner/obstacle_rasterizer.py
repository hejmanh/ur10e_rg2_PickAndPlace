#!/usr/bin/env python

"""
ROS CollisionObject Processing and Mesh Rasterization (convert geometric shapes into 2D grid obstacles)
""" 

from typing import Tuple
from .grid_2d import Grid2D, CellType


def rasterize_collision_object(grid: Grid2D, collision_msg):
    """
    Process CollisionObject message and rasterize obstacles based on object ID
    
    Args:
        grid: Grid2D instance to update
        collision_msg: moveit_msgs/CollisionObject message
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
            # For printer: rasterize only the cutting surface at z=0.1
            print(f"Printer mesh received ({collision_msg.id}) - rasterizing cutting surface at z=0.1")
            rasterize_printer_cutting_surface(mesh, grid, cutting_height=0.1, safety_margin=0.02)
            
        else:
            # For other objects: rasterize as obstacles
            print(f"Unknown object {collision_msg.id} - rasterizing as obstacle")
            rasterize_mesh_as_obstacle(mesh, grid, safety_margin=0.10)
    
    except Exception as e:
        print(f"Error processing collision object {collision_msg.id}: {e}")


def rasterize_printer_cutting_surface(ros_mesh, grid: Grid2D, cutting_height: float = 0.2, safety_margin: float = 0.02):
    """
    Rasterize only the 3D printer cutting surface at specified height 
    This creates precise obstacle representation of the actual cutting area
    
    Args:
        ros_mesh: shape_msgs/Mesh with vertices  
        grid: Grid2D instance to update
        cutting_height: Z-height of cutting surface (meters)
        safety_margin: Additional clearance around cutting area (meters)
    """
    try:
        if not ros_mesh.vertices or len(ros_mesh.vertices) == 0:
            print("Warning: Empty mesh vertices")
            return
        
        # Filter vertices at cutting height (with tolerance)
        height_tolerance = 0.02  # 2cm tolerance around cutting height
        cutting_vertices = []
        
        for vertex in ros_mesh.vertices:
            if abs(vertex.z - cutting_height) <= height_tolerance:
                cutting_vertices.append((vertex.x, vertex.y))
        
        if not cutting_vertices:
            print(f"No vertices found at cutting height {cutting_height}m")
            return
        
        print(f"Found {len(cutting_vertices)} vertices at cutting height {cutting_height}m")
        
        # Extract cutting surface bounds with more restrictive filtering
        x_coords = [v[0] for v in cutting_vertices]
        y_coords = [v[1] for v in cutting_vertices]
        
        # Filter out vertices that are likely table/pick areas (far from printer center)
        # Assume printer center is around the middle of the vertex cloud
        center_x = (min(x_coords) + max(x_coords)) / 2
        center_y = (min(y_coords) + max(y_coords)) / 2
        
        # Only include vertices within reasonable cutting area distance from center
        cutting_radius = 0.3  # 30cm radius for actual cutting area
        filtered_vertices = []
        
        for v in cutting_vertices:
            dist_from_center = ((v[0] - center_x)**2 + (v[1] - center_y)**2)**0.5
            if dist_from_center <= cutting_radius:
                filtered_vertices.append(v)
        
        if not filtered_vertices:
            print("No vertices found in central cutting area - using all vertices")
            filtered_vertices = cutting_vertices
        
        print(f"Filtered to {len(filtered_vertices)} vertices in cutting area (radius: {cutting_radius}m)")
        
        # Use filtered vertices for bounds
        x_coords = [v[0] for v in filtered_vertices]
        y_coords = [v[1] for v in filtered_vertices]
        
        min_x = min(x_coords) - safety_margin
        max_x = max(x_coords) + safety_margin
        min_y = min(y_coords) - safety_margin
        max_y = max(y_coords) + safety_margin
        
        print(f"Cutting surface bounds: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]")
        
        occupied_count = 0
        
        # Mark grid cells as occupied within cutting surface area
        for i in range(grid.width):
            for j in range(grid.height):
                world_x, world_y = grid.grid_to_world(i, j)
                
                # Check if cell is within cutting surface bounds
                if (min_x <= world_x <= max_x and min_y <= world_y <= max_y):
                    grid.set_occupied(i, j)
                    occupied_count += 1
        
        print(f"Cutting surface rasterization: {occupied_count} cells marked as occupied at z={cutting_height}m")
        
    except Exception as e:
        print(f"Error rasterizing printer cutting surface: {e}")


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


def clear_collision_object(grid: Grid2D, collision_msg):
    """
    Remove collision object from grid (set cells back to FREE)
    
    Args:
        grid: Grid2D instance to update
        collision_msg: moveit_msgs/CollisionObject message with REMOVE operation
    """
    try:
        if collision_msg.operation == collision_msg.REMOVE:
            print(f"Removing collision object: {collision_msg.id}")
            
            # to do
            pass
        
    except Exception as e:
        print(f"Error removing collision object {collision_msg.id}: {e}")


def update_collision_object_pose(grid: Grid2D, collision_msg):
    """
    Update collision object pose (MOVE operation)
    
    Args:
        grid: Grid2D instance to update
        collision_msg: moveit_msgs/CollisionObject message with MOVE operation
    """
    try:
        if collision_msg.operation == collision_msg.MOVE:
            print(f"Moving collision object: {collision_msg.id}")
            
            # require tracking object positions and updating accordingly
            pass
        
    except Exception as e:
        print(f"Error moving collision object {collision_msg.id}: {e}")