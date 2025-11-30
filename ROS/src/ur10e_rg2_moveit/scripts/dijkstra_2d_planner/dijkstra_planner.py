#!/usr/bin/env python

"""
Dijkstra 2D Path Planner 
Core Dijkstra Algorithm, Node Representation, Path Smoothing
"""

import heapq
from typing import List, Tuple, Optional
from .grid_2d import Grid2D, CellType
from .geometry_utils import bresenham_line, euclidean_distance


class GridNode:
    """
    Node representation for Dijkstra algorithm
    """
    def __init__(self, x: int, y: int, cost: float = float('inf'), parent: Optional['GridNode'] = None):
        self.x = x              # Grid x-coordinate
        self.y = y              # Grid y-coordinate
        self.cost = cost        # Distance from start
        self.parent = parent    # Parent node for path reconstruction
    
    def world_pos(self) -> Tuple[float, float]:
        """Convert to world coordinates"""
        x = (self.x * 0.05) - 1.5
        y = (self.y * 0.05) - 1.5
        return x, y
    
    def __lt__(self, other: 'GridNode') -> bool:
        """For priority queue ordering"""
        return self.cost < other.cost
    
    def __eq__(self, other: 'GridNode') -> bool:
        """For equality comparison"""
        return self.x == other.x and self.y == other.y
    
    def __hash__(self) -> int:
        """For use in sets and dictionaries"""
        return hash((self.x, self.y))


class DijkstraPlanner2D:
    """
    2D Dijkstra path planner for UR10e RG2 system
    
    Implements:
    - Section 2.1: Core Algorithm Structure
    - Section 2.2: Dijkstra Search Implementation  
    - Section 2.3: Path Smoothing
    """
    
    def __init__(self, grid: Grid2D):
        """Initialize planner with grid"""
        self.grid = grid
    
    def dijkstra_search(self, start_world: Tuple[float, float], 
                       goal_world: Tuple[float, float]) -> Tuple[Optional[List[Tuple[float, float]]], str]:
        """
        Find shortest 2D path from start to goal using Dijkstra's algorithm
        
        Args:
            start_world: (x, y) start position in world coordinates
            goal_world: (x, y) goal position in world coordinates
        
        Returns:
            Tuple of (path, error_message). Path is None if no path found.
        """
        try:
            # Convert to grid coordinates
            start_grid = self.grid.world_to_grid(*start_world)
            goal_grid = self.grid.world_to_grid(*goal_world)
            
            # Check if start is valid
            if not self.grid.is_free_cell(*start_grid):
                start_cell_type = self.grid.get_cell_type(*start_grid)
                if start_cell_type == CellType.OCCUPIED:
                    return None, f"Start position {start_world} is in occupied space (grid: {start_grid})"
                elif start_cell_type == CellType.BOUNDARY:
                    return None, f"Start position {start_world} is outside workspace boundary (grid: {start_grid})"
                else:
                    return None, f"Start position {start_world} is not in free space (grid: {start_grid})"
            
            # Check if goal is valid
            if not self.grid.is_free_cell(*goal_grid):
                goal_cell_type = self.grid.get_cell_type(*goal_grid)
                if goal_cell_type == CellType.OCCUPIED:
                    return None, f"Goal position {goal_world} is in occupied space (grid: {goal_grid})"
                elif goal_cell_type == CellType.BOUNDARY:
                    return None, f"Goal position {goal_world} is outside workspace boundary (grid: {goal_grid})"
                else:
                    return None, f"Goal position {goal_world} is not in free space (grid: {goal_grid})"
            
            print(f"Planning path from {start_world} (grid: {start_grid}) to {goal_world} (grid: {goal_grid})")
        
            # Initialize Dijkstra algorithm
            open_list = []  # Priority queue (cost, unique_id, node)
            closed_set = set()
            
            start_node = GridNode(*start_grid, cost=0.0)
            unique_id = 0
            heapq.heappush(open_list, (0.0, unique_id, start_node))
            unique_id += 1
            
            # Cost tracking
            cost_map = {start_grid: 0.0}
            
            nodes_expanded = 0
            max_nodes = 10000  # Prevent infinite loops
            
            while open_list and nodes_expanded < max_nodes:
                current_cost, _, current_node = heapq.heappop(open_list)
                current_pos = (current_node.x, current_node.y)
                
                # Skip if already processed
                if current_pos in closed_set:
                    continue
                    
                closed_set.add(current_pos)
                nodes_expanded += 1
                
                # Check if goal reached
                if current_pos == goal_grid:
                    path = self._reconstruct_path(current_node)
                    print(f"Path found! Nodes expanded: {nodes_expanded}, waypoints: {len(path)}")
                    return path, "Success"
                
                # Explore neighbors with 8-connectivity
                for neighbor in self.get_neighbors(current_node):
                    neighbor_pos = (neighbor.x, neighbor.y)
                    
                    if neighbor_pos in closed_set:
                        continue
                    
                    # Calculate new cost
                    edge_cost = self.calculate_edge_cost(current_node, neighbor)
                    new_cost = current_cost + edge_cost
                    
                    # Update if better path found
                    if neighbor_pos not in cost_map or new_cost < cost_map[neighbor_pos]:
                        cost_map[neighbor_pos] = new_cost
                        neighbor.cost = new_cost
                        neighbor.parent = current_node
                        
                        heapq.heappush(open_list, (new_cost, unique_id, neighbor))
                        unique_id += 1
            
            # Path not found
            if nodes_expanded >= max_nodes:
                error_msg = f"Search exceeded maximum nodes ({max_nodes}). Nodes expanded: {nodes_expanded}"
            else:
                error_msg = f"No path exists between start and goal. Nodes expanded: {nodes_expanded}"
            
            print(error_msg)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"Dijkstra search failed with exception: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def get_neighbors(self, node: GridNode) -> List[GridNode]:
        """
        Get valid neighboring cells (8-connectivity)
        Returns list of GridNode objects for valid neighbors
        """
        neighbors = []
        
        # 8-direction connectivity
        directions = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        
        for dx, dy in directions:
            nx, ny = node.x + dx, node.y + dy
            
            # Check bounds and if cell is free
            if self.grid.is_free_cell(nx, ny):
                neighbors.append(GridNode(nx, ny, float('inf')))
        
        return neighbors
    
    def calculate_edge_cost(self, current: GridNode, neighbor: GridNode) -> float:
        """
        Calculate movement cost between adjacent cells
        
        Returns:
            Euclidean distance in meters
        """
        dx = abs(neighbor.x - current.x)
        dy = abs(neighbor.y - current.y)
        
        # Euclidean distance in grid space converted to world units
        if dx == 1 and dy == 1:  # Diagonal movement
            return 1.414 * self.grid.resolution  # √2 * 0.05 ≈ 0.071m
        else:  # Cardinal movement
            return 1.0 * self.grid.resolution    # 0.05m
    
    def _reconstruct_path(self, goal_node: GridNode) -> List[Tuple[float, float]]:
        """Reconstruct path from goal to start using parent pointers"""
        path = []
        current = goal_node
        
        while current is not None:
            world_pos = current.world_pos()
            path.append(world_pos)
            current = current.parent
        
        path.reverse()  # Start to goal order
        return self._smooth_path(path)
    
    def _smooth_path(self, raw_path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Conservative path smoothing to preserve obstacle avoidance"""
        if len(raw_path) <= 3:
            return raw_path
        
        smoothed_path = [raw_path[0]]  # Keep start point
        
        i = 0
        while i < len(raw_path) - 1:
            max_skip = min(3, len(raw_path) - i - 1) 
            furthest_reachable = i + 1
            
            # Try to skip waypoints but be conservative near obstacles
            for j in range(i + 2, i + 1 + max_skip):
                if j >= len(raw_path):
                    break
                    
                if self._has_line_of_sight(raw_path[i], raw_path[j]):
                    # Check if path between points is safe (no close obstacles)
                    if self._is_safe_skip(raw_path[i], raw_path[j]):
                        furthest_reachable = j
                    else:
                        break  # Don't skip if near obstacles
                else:
                    break
            
            smoothed_path.append(raw_path[furthest_reachable])
            i = furthest_reachable
        
        # Ensure goal is included
        if smoothed_path[-1] != raw_path[-1]:
            smoothed_path.append(raw_path[-1])
        
        print(f"Conservative smoothing: {len(raw_path)} -> {len(smoothed_path)} waypoints")
        return smoothed_path
    
    def _has_line_of_sight(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> bool:
        """Check if direct line between two points is collision-free"""
        
        # Convert to grid coordinates
        x1, y1 = self.grid.world_to_grid(*pos1)
        x2, y2 = self.grid.world_to_grid(*pos2)
        
        # Get all points along line using Bresenham algorithm
        points = bresenham_line(x1, y1, x2, y2)
        
        # Check all points along line for collisions
        for px, py in points:
            if not self.grid.is_free_cell(px, py):
                return False
        
        return True
    
    def _is_safe_skip(self, pos1: Tuple[float, float], 
                     pos2: Tuple[float, float]) -> bool:
        """Check if it's safe to skip waypoints (no nearby obstacles)"""
        
        # Convert to grid coordinates
        x1, y1 = self.grid.world_to_grid(*pos1)
        x2, y2 = self.grid.world_to_grid(*pos2)
        
        # Get all points along line
        points = bresenham_line(x1, y1, x2, y2)
        
        # Check for nearby obstacles (1-cell margin)
        for px, py in points:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_x, check_y = px + dx, py + dy
                    if (0 <= check_x < self.grid.width and 
                        0 <= check_y < self.grid.height):
                        if self.grid.get_cell_type(check_x, check_y) == CellType.OCCUPIED:
                            return False  # Too close to obstacle
        
        return True
    
    def find_path_with_statistics(self, start_world: Tuple[float, float], 
                                 goal_world: Tuple[float, float]) -> dict:
        """
        Find path and return comprehensive statistics
        
        Returns:
            Dictionary with path, statistics, and performance metrics
        """
        import time
        
        start_time = time.time()
        path, error_message = self.dijkstra_search(start_world, goal_world)
        end_time = time.time()
        
        if path:
            path_length = sum(euclidean_distance(path[i], path[i+1]) 
                            for i in range(len(path)-1))
            
            return {
                'path': path,
                'success': True,
                'path_length': path_length,
                'waypoint_count': len(path),
                'planning_time': end_time - start_time,
                'start': start_world,
                'goal': goal_world,
                'message': error_message
            }
        else:
            return {
                'path': None,
                'success': False,
                'path_length': 0.0,
                'waypoint_count': 0,
                'planning_time': end_time - start_time,
                'start': start_world,
                'goal': goal_world,
                'message': error_message
            }
    
    def diagnose_planning_issue(self, start_world: Tuple[float, float], 
                               goal_world: Tuple[float, float]) -> str:
        """
        Diagnose why path planning might be failing
        
        Returns:
            Diagnostic message explaining potential issues
        """
        start_grid = self.grid.world_to_grid(*start_world)
        goal_grid = self.grid.world_to_grid(*goal_world)
        
        diagnostics = []
        diagnostics.append(f"Planning from world {start_world} to {goal_world}")
        diagnostics.append(f"Grid coordinates: {start_grid} to {goal_grid}")
        
        # Check grid state
        grid_info = self.grid.get_grid_info()
        diagnostics.append(f"Grid state: {grid_info['free_cells']} free, {grid_info['occupied_cells']} occupied, {grid_info['boundary_cells']} boundary")
        
        # Check start/goal accessibility
        start_cell = self.grid.get_cell_type(*start_grid)
        goal_cell = self.grid.get_cell_type(*goal_grid)
        
        cell_types = {CellType.FREE: "FREE", CellType.OCCUPIED: "OCCUPIED", CellType.BOUNDARY: "BOUNDARY"}
        diagnostics.append(f"Start cell type: {cell_types.get(start_cell, 'UNKNOWN')}")
        diagnostics.append(f"Goal cell type: {cell_types.get(goal_cell, 'UNKNOWN')}")
        
        # Check if there are any free cells at all
        if grid_info['free_cells'] == 0:
            diagnostics.append("WARNING: No free cells in grid - all space is occupied or boundary")
        
        # Check if start and goal are too close to obstacles
        start_neighbors = self._count_free_neighbors(*start_grid)
        goal_neighbors = self._count_free_neighbors(*goal_grid)
        diagnostics.append(f"Start has {start_neighbors}/8 free neighbors")
        diagnostics.append(f"Goal has {goal_neighbors}/8 free neighbors")
        
        return "\\n".join(diagnostics)
    
    def _count_free_neighbors(self, gx: int, gy: int) -> int:
        """Count how many neighboring cells are free"""
        count = 0
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dx, dy in directions:
            nx, ny = gx + dx, gy + dy
            if self.grid.is_free_cell(nx, ny):
                count += 1
        
        return count
        """Print detailed path information for debugging"""
        if not path:
            print("No path to debug")
            return
        
        print(f"\n=== PATH DEBUG INFO ===")
        print(f"Total waypoints: {len(path)}")
        
        for i, (x, y) in enumerate(path):
            gx, gy = self.grid.world_to_grid(x, y)
            print(f"  {i}: World({x:.3f}, {y:.3f}) → Grid({gx}, {gy})")
            
            if i > 0:
                prev_x, prev_y = path[i-1]
                distance = euclidean_distance((x, y), (prev_x, prev_y))
                print(f"       Distance from prev: {distance:.3f}m")
        
        print(f"=== END DEBUG INFO ===\n")
    
    def visualize_path(self, path: List[Tuple[float, float]], filename: str = None):
        """
        Create enhanced text visualization of path on grid with complete line segments
        
        Args:
            path: List of world coordinate waypoints
            filename: Optional file to save visualization
        """
        if not path:
            print("No path to visualize")
            return
        
        # Print path statistics first
        print(f"\nPath Statistics:")
        print(f"Total waypoints: {len(path)}")
        print(f"Start: {path[0]}")
        print(f"Goal: {path[-1]}")
        if len(path) > 2:
            print(f"Intermediate waypoints: {len(path) - 2}")
        
        # Create visualization grid
        vis_grid = [['.' for _ in range(self.grid.height)] for _ in range(self.grid.width)]
        
        # Mark obstacles first
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                cell_type = self.grid.get_cell_type(i, j)
                if cell_type == CellType.OCCUPIED:
                    vis_grid[i][j] = '#'
                elif cell_type == CellType.BOUNDARY:
                    vis_grid[i][j] = 'X'
        
        # Draw COMPLETE path including line segments between waypoints
        if len(path) > 1:
            path_cells_count = 0
            
            # Draw lines between consecutive waypoints
            for i in range(len(path) - 1):
                start_gx, start_gy = self.grid.world_to_grid(*path[i])
                end_gx, end_gy = self.grid.world_to_grid(*path[i + 1])
                
                # Get all points along the line using Bresenham algorithm
                line_points = bresenham_line(start_gx, start_gy, end_gx, end_gy)
                
                for px, py in line_points:
                    if 0 <= px < self.grid.width and 0 <= py < self.grid.height:
                        # Only mark as path if it's not an obstacle
                        if vis_grid[px][py] not in ['#', 'X']:
                            vis_grid[px][py] = '*'
                            path_cells_count += 1
            
            print(f"Path cells marked: {path_cells_count}")
        
        # Mark waypoints (will overwrite some '*' symbols)
        for i, waypoint in enumerate(path):
            gx, gy = self.grid.world_to_grid(*waypoint)
            if 0 <= gx < self.grid.width and 0 <= gy < self.grid.height:
                if i == 0:
                    vis_grid[gx][gy] = 'S'  # Start
                elif i == len(path) - 1:
                    vis_grid[gx][gy] = 'G'  # Goal
                else:
                    vis_grid[gx][gy] = 'W'  # Waypoint
        
        # Print grid visualization with coordinates
        print(f"\nGrid Visualization:")
        print("S=Start, G=Goal, W=Waypoint, *=Path, #=Obstacle, X=Boundary, .=Free\n")
        
        # Print column indices
        print("   ", end="")
        for i in range(0, self.grid.width, 10):
            print(f"{i:10d}", end="")
        print()
        
        for j in range(self.grid.height-1, -1, -1):  # Top to bottom
            # Print Y coordinate every 10 rows
            if j % 10 == 0:
                print(f"{j:2d} ", end="")
            else:
                print("   ", end="")
            
            row = ''.join(vis_grid[i][j] for i in range(self.grid.width))
            print(row)
        
        # Count visualization elements
        path_count = sum(row.count('*') for row in [''.join(vis_grid[i][j] for i in range(self.grid.width)) for j in range(self.grid.height)])
        obstacle_count = sum(row.count('#') for row in [''.join(vis_grid[i][j] for i in range(self.grid.width)) for j in range(self.grid.height)])
        boundary_count = sum(row.count('X') for row in [''.join(vis_grid[i][j] for i in range(self.grid.width)) for j in range(self.grid.height)])
        
        print(f"\nVisualization Summary:")
        print(f"  Path cells (*): {path_count}")
        print(f"  Obstacles (#): {obstacle_count}")
        print(f"  Boundaries (X): {boundary_count}")
        print(f"  Waypoints (S,G,W): {len(path)}")
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(f"Path Statistics:\n")
                    f.write(f"Total waypoints: {len(path)}\n")
                    f.write(f"Start: {path[0]}\n")
                    f.write(f"Goal: {path[-1]}\n\n")
                    f.write("Path Visualization:\n")
                    f.write("S=Start, G=Goal, W=Waypoint, *=Path, #=Obstacle, X=Boundary, .=Free\n\n")
                    
                    # Write grid with coordinates
                    f.write("   ")
                    for i in range(0, self.grid.width, 10):
                        f.write(f"{i:10d}")
                    f.write("\n")
                    
                    for j in range(self.grid.height-1, -1, -1):
                        if j % 10 == 0:
                            f.write(f"{j:2d} ")
                        else:
                            f.write("   ")
                        row = ''.join(vis_grid[i][j] for i in range(self.grid.width))
                        f.write(row + '\n')
                    
                    f.write(f"\nVisualization Summary:\n")
                    f.write(f"  Path cells (*): {path_count}\n")
                    f.write(f"  Obstacles (#): {obstacle_count}\n")
                    f.write(f"  Boundaries (X): {boundary_count}\n")
                    f.write(f"  Waypoints (S,G,W): {len(path)}\n")
                    
                print(f"Visualization saved to {filename}")
            except Exception as e:
                print(f"Failed to save visualization: {e}")