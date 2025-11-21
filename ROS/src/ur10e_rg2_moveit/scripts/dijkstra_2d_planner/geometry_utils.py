#!/usr/bin/env python

"""
Geometry Utilities
Convex Hull, Polygon Operations, Point-in-Polygon Testing
"""

import math
from typing import List, Tuple


def compute_convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Compute convex hull of a set of 2D points using Graham scan algorithm
    
    Args:
        points: List of (x, y) tuples
    
    Returns:
        List of (x, y) tuples representing convex hull vertices in counter-clockwise order
    """
    def cross_product(o, a, b):
        """
        Calculate cross product of vectors OA and OB
        - cross product > 0: counter-clockwise turn
        - cross product < 0: clockwise turn
        - cross product = 0: collinear points
        """
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    if len(points) < 3:
        return points
    
    # Sort points lexicographically
    points = sorted(set(points))
    
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Remove last point of each half because it's repeated
    return lower[:-1] + upper[:-1]


def expand_polygon(polygon: List[Tuple[float, float]], margin: float) -> List[Tuple[float, float]]:
    """
    Expand polygon outward by specified margin using offset algorithm
    
    Args:
        polygon: List of (x, y) tuples representing polygon vertices
        margin: Distance to expand polygon outward (meters)
    
    Returns:
        List of (x, y) tuples representing expanded polygon
    """
    if not polygon or len(polygon) < 3:
        return polygon
    
    expanded_vertices = []
    n = len(polygon)
    
    for i in range(n):
        # Get current vertex and adjacent vertices
        prev_vertex = polygon[(i - 1) % n]
        current_vertex = polygon[i]
        next_vertex = polygon[(i + 1) % n]
        
        # Calculate edge vectors
        edge1 = (current_vertex[0] - prev_vertex[0], current_vertex[1] - prev_vertex[1])
        edge2 = (next_vertex[0] - current_vertex[0], next_vertex[1] - current_vertex[1])
        
        # Calculate edge lengths
        len1 = math.sqrt(edge1[0]**2 + edge1[1]**2)
        len2 = math.sqrt(edge2[0]**2 + edge2[1]**2)
        
        if len1 == 0 or len2 == 0:
            continue
        
        # Normalize edge vectors
        norm1 = (edge1[0]/len1, edge1[1]/len1)
        norm2 = (edge2[0]/len2, edge2[1]/len2)
        
        # Calculate perpendicular vectors (pointing outward)
        perp1 = (-norm1[1], norm1[0])
        perp2 = (-norm2[1], norm2[0])
        
        # Calculate angle bisector
        bisector = (perp1[0] + perp2[0], perp1[1] + perp2[1])
        bisector_len = math.sqrt(bisector[0]**2 + bisector[1]**2)
        
        if bisector_len > 0:
            bisector = (bisector[0]/bisector_len, bisector[1]/bisector_len)
            
            # Calculate offset distance (accounting for angle)
            dot_product = norm1[0]*norm2[0] + norm1[1]*norm2[1]
            angle_factor = 1.0 / math.sqrt((1 + dot_product) / 2) if dot_product > -0.999 else margin
            offset_distance = margin * angle_factor
            
            # Calculate expanded vertex
            expanded_x = current_vertex[0] + bisector[0] * offset_distance
            expanded_y = current_vertex[1] + bisector[1] * offset_distance
            expanded_vertices.append((expanded_x, expanded_y))
    
    return expanded_vertices


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Test if point is inside polygon using ray casting algorithm
    
    Args:
        point: (x, y) tuple to test
        polygon: List of (x, y) tuples representing polygon vertices
    
    Returns:
        True if point is inside polygon
    """
    if not polygon:
        return False
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def bresenham_line(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    """
    Generate points along line using Bresenham's algorithm
    Used for line-of-sight collision checking in pathfinding
    
    Args:
        x1, y1: Start point coordinates
        x2, y2: End point coordinates
    
    Returns:
        List of (x, y) integer coordinate points along the line
    """
    points = []
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    
    points.append((x2, y2))
    return points


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
    
    Returns:
        Distance in same units as input coordinates
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two grid points
    
    Args:
        p1: First grid point (x, y)
        p2: Second grid point (x, y)
    
    Returns:
        Manhattan distance as integer
    """
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])