#!/usr/bin/env python

"""
Test Suite for Dijkstra 2D Planner Module
Comprehensive testing and examples for UR10e RG2 Pick-and-Place System
"""

import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module components
from dijkstra_2d_planner import (
    Grid2D, CellType, DijkstraPlanner2D, 
    create_planner_system, get_module_info,
    rasterize_collision_object, rasterize_workspace_boundaries
)


def test_grid_specifications():
    """Test Section 1.1: Grid Specifications"""
    print("=" * 60)
    print("Testing Grid Specifications (Section 1.1)")
    print("=" * 60)
    
    grid = Grid2D()
    info = grid.get_grid_info()
    
    print(f"Grid Size: {info['width']} × {info['height']} = {info['total_cells']} cells")
    print(f"Resolution: {info['resolution']}m per cell")
    print(f"World bounds: X[{info['world_bounds']['x_min']}, {info['world_bounds']['x_max']}], Y[{info['world_bounds']['y_min']}, {info['world_bounds']['y_max']}]")
    
    # Test coordinate conversions
    test_points = [
        (0.0, 0.0),      # Center (robot base)
        (-1.5, -1.5),    # Bottom-left corner
        (1.5, 1.5),      # Top-right corner
        (0.75, -0.75),   # Random point
        (-0.5, 1.0)      # Another test point
    ]
    
    print("\nCoordinate System Integration:")
    for x, y in test_points:
        grid_x, grid_y = grid.world_to_grid(x, y)
        x_back, y_back = grid.grid_to_world(grid_x, grid_y)
        print(f"  World ({x:5.2f}, {y:5.2f}) → Grid ({grid_x:2d}, {grid_y:2d}) → World ({x_back:5.2f}, {y_back:5.2f})")
    
    print("Grid specifications test passed")
    return True


def test_cell_classification():
    """Test Section 1.2: Grid Cell Classification"""
    print("\n" + "=" * 60)
    print("Testing Grid Cell Classification (Section 1.2)")
    print("=" * 60)
    
    grid = Grid2D()
    
    # Test cell type operations
    grid.set_occupied(30, 30)  # Center area
    grid.set_occupied(50, 40)  # Some obstacle
    grid.set_boundary(0, 0)    # Corner as boundary
    
    test_cells = [(30, 30), (50, 40), (0, 0), (20, 20)]
    cell_type_names = {CellType.FREE: "FREE", CellType.OCCUPIED: "OCCUPIED", CellType.BOUNDARY: "BOUNDARY"}
    
    print("Cell Classification Test:")
    for gx, gy in test_cells:
        cell_type = grid.get_cell_type(gx, gy)
        wx, wy = grid.grid_to_world(gx, gy)
        print(f"  Grid ({gx:2d}, {gy:2d}) = World ({wx:5.2f}, {wy:5.2f}) → {cell_type_names[cell_type]}")
    
    print("Cell classification test passed")
    return True


def test_obstacle_rasterization():
    """Test Section 1.3: Obstacle Rasterization"""
    print("\n" + "=" * 60)
    print("Testing Obstacle Rasterization (Section 1.3)")
    print("=" * 60)
    
    grid = Grid2D()
    
    # Create mock ROS collision object for testing
    class MockCollisionObject:
        ADD = 0
        
        def __init__(self, object_id, mesh_vertices):
            self.id = object_id
            self.operation = self.ADD
            self.meshes = [MockRosMesh(mesh_vertices)]
    
    class MockRosMesh:
        def __init__(self, vertex_coords):
            self.vertices = [MockRosPoint(x, y, z) for x, y, z in vertex_coords]
    
    class MockRosPoint:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    # Test workspace boundary marking
    print("Workspace boundary marking:")
    rasterize_workspace_boundaries(grid, safety_margin=0.05)
    
    # Test printer collision object
    printer_vertices = [
        (0.6, 0.6, 0.0), (1.1, 0.6, 0.0),
        (1.1, 1.1, 0.0), (0.6, 1.1, 0.0)
    ]
    printer_collision = MockCollisionObject("3d_printer", printer_vertices)
    
    print("ROS collision object processing:")
    rasterize_collision_object(printer_collision, grid)
    
    # Display final statistics
    final_info = grid.get_grid_info()
    print(f"\nFinal Grid Statistics:")
    print(f"  Free cells: {final_info['free_cells']}")
    print(f"  Occupied cells: {final_info['occupied_cells']}")
    print(f"  Boundary cells: {final_info['boundary_cells']}")
    print(f"  Occupied percentage: {final_info['occupied_cells']/final_info['total_cells']*100:.1f}%")
    
    print("Obstacle rasterization test passed")
    return True


def test_dijkstra_pathfinding():
    """Test Dijkstra pathfinding algorithm"""
    print("\n" + "=" * 60)
    print("Testing Dijkstra Pathfinding Algorithm")
    print("=" * 60)
    
    # Create grid with some obstacles
    grid, planner = create_planner_system(setup_obstacles=True)
    
    # Add a simple obstacle in the middle
    for i in range(25, 35):
        for j in range(25, 35):
            grid.set_occupied(i, j)
    
    print("Grid setup with central obstacle complete")
    
    # Test pathfinding
    start = (-1.0, -1.0)  # Bottom-left area
    goal = (1.0, 1.0)     # Top-right area
    
    print(f"Planning path from {start} to {goal}")
    
    result = planner.find_path_with_statistics(start, goal)
    
    if result['success']:
        print(f"Path found!")
        print(f"  Path length: {result['path_length']:.3f}m")
        print(f"  Waypoints: {result['waypoint_count']}")
        print(f"  Planning time: {result['planning_time']:.3f}s")
        
        # Show first few waypoints
        path = result['path']
        print(f"  First 3 waypoints: {path[:3]}")
        print(f"  Last 3 waypoints: {path[-3:]}")
        
        # Create visualization
        planner.debug_path_details(path)  # Show detailed path info
        planner.visualize_path(path)      # Show enhanced visualization
        
    else:
        print("No path found")
        return False
    
    print("Dijkstra pathfinding test passed")
    return True


def test_module_integration():
    """Test complete module integration"""
    print("\n" + "=" * 60)
    print("Testing Complete Module Integration")
    print("=" * 60)
    
    # Test module info
    info = get_module_info()
    print(f"Module: {info['name']} v{info['version']}")
    print(f"Description: {info['description']}")
    
    print("\nModule Components:")
    for component in info['components']:
        print(f"  • {component}")
    
    print("\nGrid Specifications:")
    for key, value in info['grid_specs'].items():
        print(f"  {key}: {value}")
    
    # Test convenience function
    grid, planner = create_planner_system(setup_obstacles=False)
    
    print("\nTesting convenience function:")
    print(f"  Grid created: {type(grid).__name__}")
    print(f"  Planner created: {type(planner).__name__}")
    print(f"  Grid size: {grid.width}x{grid.height}")
    
    print("Module integration test passed")
    return True


def run_performance_test():
    """Run performance test with larger pathfinding scenarios"""
    print("\n" + "=" * 60)
    print("Performance Test - Multiple Path Queries")
    print("=" * 60)
    
    grid, planner = create_planner_system(setup_obstacles=True)
    
    # Add some scattered obstacles
    import random
    random.seed(42)  # Reproducible results
    
    obstacle_count = 0
    for _ in range(200):  # Add 200 random obstacles
        x, y = random.randint(5, 54), random.randint(5, 54)
        if grid.is_free_cell(x, y):
            grid.set_occupied(x, y)
            obstacle_count += 1
    
    print(f"Added {obstacle_count} random obstacles")
    
    # Test multiple path queries
    test_cases = [
        ((-1.0, -1.0), (1.0, 1.0)),   # Diagonal
        ((-1.0, 1.0), (1.0, -1.0)),   # Other diagonal
        ((0.0, -1.0), (0.0, 1.0)),    # Vertical
        ((-1.0, 0.0), (1.0, 0.0)),    # Horizontal
        ((-0.5, -0.5), (0.5, 0.5))    # Shorter path
    ]
    
    total_time = 0
    successful_paths = 0
    
    for i, (start, goal) in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {start} → {goal}")
        result = planner.find_path_with_statistics(start, goal)
        
        if result['success']:
            successful_paths += 1
            total_time += result['planning_time']
            print(f"Success: {result['path_length']:.3f}m, {result['waypoint_count']} waypoints, {result['planning_time']:.3f}s")
        else:
            print(f"Failed")
    
    print(f"\nPerformance Summary:")
    print(f"  Successful paths: {successful_paths}/{len(test_cases)}")
    print(f"  Average planning time: {total_time/max(successful_paths,1):.3f}s")
    print(f"  Total planning time: {total_time:.3f}s")
    
    print("Performance test completed")
    return True


def main():
    """Run all tests"""
    print("Dijkstra 2D Planner Module - Test Suite")
    print("UR10e RG2 Pick-and-Place System")
    print("=" * 80)
    
    tests = [
        test_grid_specifications,
        test_cell_classification, 
        test_obstacle_rasterization,
        test_dijkstra_pathfinding,
        test_module_integration,
        run_performance_test
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Test {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\nALL TESTS PASSED! Module is ready for integration")
    else:
        print(f"\n  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)