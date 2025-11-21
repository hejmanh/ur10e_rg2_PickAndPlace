#!/usr/bin/env python

"""
2D Grid Workspace Design and Cell Classification
"""

from typing import Tuple


class CellType:
    """Grid cell classification constants"""
    FREE = 0        # Safe navigation space
    OCCUPIED = 1    # Collision with obstacles
    BOUNDARY = 2    # Grid boundaries/unreachable areas


class Grid2D:
    """
    2D Grid Workspace for UR10e RG2 system
    
    Grid Specifications:
    - Planning Grid: 3.0m × 3.0m (matches Unity table surface size)
    - Robot Base Position: Center of grid (0, 0 in ROS coordinates)
    - Grid Bounds: X-axis: [-1.5m, +1.5m], Y-axis: [-1.5m, +1.5m]
    - Grid Resolution: 5cm (0.05m) per cell
    - Grid Size: 60 × 60 = 3,600 cells
    
    Cell Classification:
    - FREE: Safe navigation space
    - OCCUPIED: Collision with obstacles  
    - BOUNDARY: Grid boundaries/unreachable areas
    """
    
    def __init__(self):
        """Initialize 2D grid with specifications from plan"""
        # Grid specifications
        self.width = 60                    # 60 cells across
        self.height = 60                   # 60 cells deep
        self.resolution = 0.05             # 5cm per cell
        self.origin = (-1.5, -1.5)        # Bottom-left corner in world coordinates
        
        # World bounds for validation
        self.min_x = -1.5
        self.max_x = 1.5
        self.min_y = -1.5
        self.max_y = 1.5
        
        # Initialize grid with all cells as FREE
        self.grid = [[CellType.FREE for _ in range(self.height)] for _ in range(self.width)]
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert ROS world coordinates to grid indices
        
        Maps world coordinates:
        X: [-1.5m, +1.5m] -> Grid X: [0, 59]
        Y: [-1.5m, +1.5m] -> Grid Y: [0, 59]
        
        Args:
            x: ROS world X coordinate (forward/backward from robot base)
            y: ROS world Y coordinate (left/right from robot base)
        
        Returns:
            (grid_x, grid_y) indices with bounds checking
        """
        grid_x = int((x + 1.5) / self.resolution)  # Map [-1.5, 1.5] to [0, 59]
        grid_y = int((y + 1.5) / self.resolution)  # Map [-1.5, 1.5] to [0, 59]
        
        # Apply bounds checking
        grid_x = max(0, min(59, grid_x))
        grid_y = max(0, min(59, grid_y))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        Convert grid indices to ROS world coordinates
        
        Maps grid indices:
        Grid X: [0, 59] -> X: [-1.5m, +1.5m]
        Grid Y: [0, 59] -> Y: [-1.5m, +1.5m]
        
        Args:
            grid_x: Grid X index
            grid_y: Grid Y index
        
        Returns:
            (x, y) world coordinates
        """
        x = (grid_x * self.resolution) - 1.5
        y = (grid_y * self.resolution) - 1.5
        return x, y
    
    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if grid coordinates are within valid bounds
        
        Args:
            grid_x: Grid X index
            grid_y: Grid Y index
        
        Returns:
            True if coordinates are within grid bounds
        """
        return 0 <= grid_x < self.width and 0 <= grid_y < self.height
    
    def is_free_cell(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if cell is free for navigation
        
        Args:
            grid_x: Grid X index
            grid_y: Grid Y index
        
        Returns:
            True if cell is FREE and within bounds
        """
        if not self.is_valid_cell(grid_x, grid_y):
            return False
        return self.grid[grid_x][grid_y] == CellType.FREE
    
    def set_cell_type(self, grid_x: int, grid_y: int, cell_type: int):
        """
        Set the type of a grid cell
        
        Args:
            grid_x: Grid X index
            grid_y: Grid Y index
            cell_type: CellType constant (FREE, OCCUPIED, BOUNDARY)
        """
        if self.is_valid_cell(grid_x, grid_y):
            self.grid[grid_x][grid_y] = cell_type
    
    def set_occupied(self, grid_x: int, grid_y: int):
        """Mark cell as occupied"""
        self.set_cell_type(grid_x, grid_y, CellType.OCCUPIED)
    
    def set_free(self, grid_x: int, grid_y: int):
        """Mark cell as free"""
        self.set_cell_type(grid_x, grid_y, CellType.FREE)
    
    def set_boundary(self, grid_x: int, grid_y: int):
        """Mark cell as boundary"""
        self.set_cell_type(grid_x, grid_y, CellType.BOUNDARY)
    
    def get_cell_type(self, grid_x: int, grid_y: int) -> int:
        """
        Get the type of a grid cell
        
        Args:
            grid_x: Grid X index
            grid_y: Grid Y index
        
        Returns:
            CellType constant or BOUNDARY if out of bounds
        """
        if not self.is_valid_cell(grid_x, grid_y):
            return CellType.BOUNDARY
        return self.grid[grid_x][grid_y]
    
    def is_world_position_valid(self, x: float, y: float) -> bool:
        """
        Check if world position is within workspace bounds
        
        Args:
            x: World X coordinate
            y: World Y coordinate
        
        Returns:
            True if position is within [-1.5, 1.5] bounds
        """
        return (self.min_x <= x <= self.max_x and 
                self.min_y <= y <= self.max_y)
    
    def get_grid_info(self) -> dict:
        """
        Get comprehensive grid information for debugging
        
        Returns:
            Grid specifications and statistics
        """
        total_cells = self.width * self.height
        free_cells = sum(1 for row in self.grid for cell in row if cell == CellType.FREE)
        occupied_cells = sum(1 for row in self.grid for cell in row if cell == CellType.OCCUPIED)
        boundary_cells = sum(1 for row in self.grid for cell in row if cell == CellType.BOUNDARY)
        
        return {
            'width': self.width,
            'height': self.height,
            'resolution': self.resolution,
            'origin': self.origin,
            'world_bounds': {
                'x_min': self.min_x,
                'x_max': self.max_x,
                'y_min': self.min_y,
                'y_max': self.max_y
            },
            'total_cells': total_cells,
            'free_cells': free_cells,
            'occupied_cells': occupied_cells,
            'boundary_cells': boundary_cells,
        }
    
    def clear_grid(self):
        """Reset all cells to FREE state"""
        for i in range(self.width):
            for j in range(self.height):
                self.grid[i][j] = CellType.FREE


def world_to_grid(x: float, y: float) -> Tuple[int, int]:
    """
    Convert ROS world coordinates to grid indices
    Standalone utility function as specified in plan
    
    Args:
        x: World X coordinate
        y: World Y coordinate
    
    Returns:
        (grid_x, grid_y) indices
    """
    grid_x = int((x + 1.5) / 0.05)  # Map [-1.5, 1.5] to [0, 59]
    grid_y = int((y + 1.5) / 0.05)  # Map [-1.5, 1.5] to [0, 59]
    return max(0, min(59, grid_x)), max(0, min(59, grid_y))


def grid_to_world(grid_x: int, grid_y: int) -> Tuple[float, float]:
    """
    Convert grid indices to ROS world coordinates
    Standalone utility function as specified in plan
    
    Args:
        grid_x: Grid X index
        grid_y: Grid Y index
    
    Returns:
        (x, y) world coordinates
    """
    x = (grid_x * 0.05) - 1.5
    y = (grid_y * 0.05) - 1.5
    return x, y