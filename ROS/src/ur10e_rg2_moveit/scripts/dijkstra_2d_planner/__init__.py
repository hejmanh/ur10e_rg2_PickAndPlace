#!/usr/bin/env python

"""
Dijkstra 2D Planner Module
Clean public interface for UR10e RG2 Pick-and-Place System

Provides:
- Grid2D: 2D workspace grid with coordinate conversion
- CellType: Grid cell classification constants
- DijkstraPlanner2D: Path planning algorithm
- Obstacle rasterization functions for ROS integration

Usage:
    from dijkstra_2d_planner import Grid2D, DijkstraPlanner2D, CellType
    from dijkstra_2d_planner import rasterize_collision_object
"""

# Core grid components
from .grid_2d import Grid2D, CellType, world_to_grid, grid_to_world

# Path planning algorithm  
from .dijkstra_planner import DijkstraPlanner2D, GridNode

# Obstacle processing for ROS integration
from .obstacle_rasterizer import (
    rasterize_collision_object,
    rasterize_mesh_as_obstacle,
    rasterize_workspace_boundaries,
    rasterize_rectangular_area,
    rasterize_circular_area
)

# Geometry utilities 
from .geometry_utils import (
    compute_convex_hull,
    expand_polygon, 
    point_in_polygon,
    bresenham_line,
    euclidean_distance
)

# Public API - what gets imported with "from dijkstra_2d_planner import *"
__all__ = [
    # Core classes
    'Grid2D',
    'CellType', 
    'DijkstraPlanner2D',
    'GridNode',
    
    # Coordinate conversion
    'world_to_grid',
    'grid_to_world',
    
    # Obstacle processing
    'rasterize_collision_object',
    'rasterize_workspace_boundaries',
    
    # Geometry utilities (most commonly needed)
    'euclidean_distance',
    'bresenham_line'
]

def create_planner_system(setup_obstacles: bool = True) -> tuple:
    """
    Convenience function to create a complete planner system
    
    Args:
        setup_obstacles: Whether to set up workspace boundaries
    
    Returns:
        (grid, planner) tuple ready for use
    """
    # Create grid and planner
    grid = Grid2D()
    planner = DijkstraPlanner2D(grid)
    
    if setup_obstacles:
        # Set up basic workspace boundaries
        rasterize_workspace_boundaries(grid, safety_margin=0.05)
        print("Planner system created with workspace boundaries")
    else:
        print("Planner system created (no obstacles)")
    
    return grid, planner

def get_module_info() -> dict:
    """
    Get information about this module
    
    Returns:
        Module information dictionary
    """
    return {
        'name': 'dijkstra_2d_planner',
        'description': '2D Dijkstra path planner for UR10e RG2 robot system',
        'components': [
            'Grid2D - 60x60 workspace grid with 5cm resolution',
            'DijkstraPlanner2D - Optimal path planning algorithm',
            'ROS CollisionObject integration',
            'Path smoothing and optimization',
            'Visualization tools'
        ],
        'grid_specs': {
            'size': '60x60 cells (3600 total)',
            'resolution': '0.05m (5cm) per cell',
            'workspace': '3.0m x 3.0m',
            'bounds': 'X[-1.5, 1.5], Y[-1.5, 1.5]',
        }
    }