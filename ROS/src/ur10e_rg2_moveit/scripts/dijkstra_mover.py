#!/usr/bin/env python

from __future__ import print_function

import os
import sys

# Add the script directory to Python path for dijkstra_2d_planner import
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import rospy
import copy
import math
import moveit_commander
import numpy as np
import time

import moveit_msgs.msg
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, BoundingVolume
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, CollisionObject
import geometry_msgs.msg
from geometry_msgs.msg import Quaternion, Pose
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from ur10e_rg2_moveit.srv import MoverService, MoverServiceRequest, MoverServiceResponse

from dijkstra_2d_planner import (
    Grid2D,
    DijkstraPlanner2D,
    CellType,
    rasterize_collision_object,
    rasterize_workspace_boundaries
)

joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

# Between Melodic and Noetic, the return type of plan() changed
if sys.version_info >= (3, 0):
    def planCompat(plan):
        return plan[1]
else:
    def planCompat(plan):
        return plan


class DijkstraMoveitServer:
    """
    Enhanced moveit server that integrates 2D Dijkstra Cartesian path planning preprocessing
    with traditional MoveIt joint space path planning.
    """
    
    def __init__(self):
        """Initialize the Dijkstra-enhanced moveit server"""
        rospy.loginfo("Initializing Dijkstra Moveit Server...")
        
        # Initialize the grid and planner components
        self.grid = Grid2D()  
        
        self.planner = DijkstraPlanner2D(self.grid)
        
        # Workspace boundaries (UR10e robot working area)
        self.workspace_bounds = {
            'x_min': -1.5, 'x_max': 1.5,
            'y_min': -1.5, 'y_max': 1.5,
            'z_min': 0.0, 'z_max': 2.0
        }
        
        # Track collision objects for path planning
        self.collision_objects = {}
        self.last_grid_update = time.time()
        
        # Configuration - Read from ROS parameters with defaults
        self.use_dijkstra_preprocessing = rospy.get_param('~use_dijkstra_preprocessing', True)
        # Height settings for 2D path planning
        # Unity sends poses with z=0 at table surface, so 10cm above table is reasonable for gripper clearance
        self.dijkstra_height = rospy.get_param('~dijkstra_height', 0.2)  # Height above table surface for 2D path planning
        self.height_tolerance = rospy.get_param('~height_tolerance', 0.05)  # Tolerance for considering poses at same height
        
        # Motion planning parameters
        self.planner_id = rospy.get_param('~planner_id', 'RRTConnect')
        self.planning_time = rospy.get_param('~planning_time', 15.0)
        self.max_planning_attempts = rospy.get_param('~max_planning_attempts', 100)
        self.waypoint_sampling_step = rospy.get_param('~waypoint_sampling_step', 10)
        #if adjust the division too high -> smaller step size -> more waypoints -> too close, RRT can not find a path
        #if adjust the division too low -> larger step size -> less waypoints -> dijkstra value lost
        
        # Update workspace bounds from parameters if provided
        self.workspace_bounds = {
            'x_min': rospy.get_param('~workspace_x_min', -1.5),
            'x_max': rospy.get_param('~workspace_x_max', 1.5),
            'y_min': rospy.get_param('~workspace_y_min', -1.5),
            'y_max': rospy.get_param('~workspace_y_max', 1.5),
            'z_min': rospy.get_param('~workspace_z_min', 0.0),
            'z_max': rospy.get_param('~workspace_z_max', 2.0)
        }
        
        # Safety parameters
        self.safety_margin = rospy.get_param('~safety_margin', 0.05)
        
        rospy.loginfo("Dijkstra Moveit Server initialized successfully!")
        rospy.loginfo("Configuration: height=%.3fm, tolerance=%.3fm, preprocessing=%s", 
                     self.dijkstra_height, self.height_tolerance, self.use_dijkstra_preprocessing)
    
    def collision_callback(self, msg):
        """
        Enhanced collision callback that updates the grid for path planning
        """
        rospy.loginfo("Received collision object: %s", msg.id)
        
        # Store the collision object
        self.collision_objects[msg.id] = msg
        
        # Log pose information if available
        if len(msg.mesh_poses) > 0:
            pose = msg.mesh_poses[0]
            rospy.loginfo("Pose: position - x=%.3f, y=%.3f, z=%.3f", 
                         pose.position.x, pose.position.y, pose.position.z)
        
        # Update the grid with new obstacle information
        self._update_grid()
    
    def _update_grid(self):
        """Update the 2D grid with current collision objects"""
        try:
            # Reset grid
            self.grid.clear_grid()
            
            # Add workspace boundaries
            rasterize_workspace_boundaries(self.grid, safety_margin=0.05)
            
            # Process each collision object
            for obj_id, collision_obj in self.collision_objects.items():
                try:
                    rasterize_collision_object(self.grid, collision_obj)
                    rospy.logdebug("Successfully rasterized collision object: %s", obj_id)
                except Exception as e:
                    rospy.logwarn("Failed to rasterize collision object %s: %s", obj_id, e)
            
            self.last_grid_update = time.time()
            rospy.loginfo("Grid updated with %d collision objects", len(self.collision_objects))
            
        except Exception as e:
            rospy.logerr("Failed to update grid: %s", e)
    
    def _poses_at_same_height(self, pose1, pose2): 
        """Check if two poses are approximately at the same height"""
        return abs(pose1.position.z - pose2.position.z) < self.height_tolerance
    
    def _pose_in_2d_workspace(self, pose):
        """Check if a pose is within the 2D workspace for Dijkstra planning"""
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        return (self.workspace_bounds['x_min'] <= x <= self.workspace_bounds['x_max'] and
                self.workspace_bounds['y_min'] <= y <= self.workspace_bounds['y_max'] and
                abs(z - self.dijkstra_height) < self.height_tolerance)
    
    def _plan_trajectory_with_dijkstra(self, move_group, destination_pose, start_joint_angles, planning_pose=None):
        """
        Enhanced trajectory planning that uses Dijkstra preprocessing when appropriate
        
        Args:
            planning_pose: Optional pose for 2D planning (at dijkstra_height)
            destination_pose: Final precise destination pose
        """
        # Use planning_pose for 2D pathfinding if provided, otherwise use destination_pose
        pose_for_2d_planning = planning_pose if planning_pose is not None else destination_pose
        
        # Get current end effector pose
        move_group.set_joint_value_target(start_joint_angles)
        current_pose = move_group.get_current_pose().pose
        
        # Check if we should use Dijkstra preprocessing
        use_dijkstra = (self.use_dijkstra_preprocessing and 
                       self._poses_at_same_height(current_pose, pose_for_2d_planning) and
                       self._pose_in_2d_workspace(current_pose) and 
                       self._pose_in_2d_workspace(pose_for_2d_planning))
        
        if use_dijkstra:
            rospy.loginfo("Using Dijkstra preprocessing for trajectory planning")
            return self._plan_with_dijkstra_preprocessing(move_group, destination_pose, 
                                                        start_joint_angles, current_pose, pose_for_2d_planning)
        else:
            rospy.loginfo("Using direct MoveIt! planning")
            return self._plan_direct_moveit(move_group, destination_pose, start_joint_angles)
    
    def _plan_with_dijkstra_preprocessing(self, move_group, destination_pose, 
                                        start_joint_angles, current_pose, planning_pose=None):
        """Plan trajectory using Dijkstra preprocessing for obstacle avoidance"""
        try:
            # Use planning_pose for 2D pathfinding if provided, otherwise use destination_pose
            pose_for_pathfinding = planning_pose if planning_pose is not None else destination_pose
            
            # Convert poses to world coordinates for planning
            start_x, start_y = current_pose.position.x, current_pose.position.y
            goal_x, goal_y = pose_for_pathfinding.position.x, pose_for_pathfinding.position.y
            
            # Verify coordinates are within grid bounds
            start_grid = self.grid.world_to_grid(start_x, start_y)
            goal_grid = self.grid.world_to_grid(goal_x, goal_y)
            
            if start_grid is None or goal_grid is None:
                rospy.logwarn("Start or goal outside grid bounds, falling back to direct planning")
                return self._plan_direct_moveit(move_group, destination_pose, start_joint_angles)
            
            # Find 2D path using Dijkstra (with world coordinates)
            rospy.loginfo("Finding 2D path from world (%.3f,%.3f) to (%.3f,%.3f)", 
                         start_x, start_y, goal_x, goal_y)
            
            path_result = self.planner.find_path_with_statistics(
                (start_x, start_y), (goal_x, goal_y))
            
            if not path_result['success']:
                # Log detailed diagnostic information
                diagnostics = self.planner.diagnose_planning_issue(
                    (start_x, start_y), (goal_x, goal_y))
                rospy.logwarn("Dijkstra planning failed:")
                for line in diagnostics.split('\\n'):
                    rospy.logwarn("  %s", line)
                rospy.logwarn("Error: %s", path_result.get('message', 'Unknown error'))
                rospy.logwarn("Falling back to direct planning")
                return self._plan_direct_moveit(move_group, destination_pose, start_joint_angles)
            
            # Path is already in world coordinates
            world_path = path_result['path']
            
            rospy.loginfo("Found 2D path with %d waypoints in %.3fs", 
                         len(world_path), path_result['planning_time'])
            
            # Plan trajectory through waypoints
            return self._plan_through_waypoints(move_group, world_path, destination_pose, 
                                              start_joint_angles)
            
        except Exception as e:
            rospy.logerr("Dijkstra preprocessing failed: %s. Falling back to direct planning", e)
            return self._plan_direct_moveit(move_group, destination_pose, start_joint_angles)
    
    def _plan_through_waypoints(self, move_group, world_path, final_pose, start_joint_angles):
        """Plan trajectory through a series of 2D waypoints"""
        
        # For very simple paths, use direct planning
        if len(world_path) <= 2:
            return self._plan_direct_moveit(move_group, final_pose, start_joint_angles)
        
        try:
            # Use the smoothed path directly (path smoothing already optimized waypoint count)
            # Only do light sampling for extremely long paths to preserve Dijkstra value
            if len(world_path) > 15:  
                step = max(1, len(world_path) // self.waypoint_sampling_step)  
                sampled_waypoints = world_path[::step]
                
                # Ensure including the final point
                if sampled_waypoints[-1] != world_path[-1]:
                    sampled_waypoints.append(world_path[-1])
            else:
                # For shorter paths, use every other waypoint to reduce density
                sampled_waypoints = world_path[::2] if len(world_path) > 5 else world_path
            
            # Skip initial waypoints to prevent table collision during pre-grasp
            # Get current end effector position to calculate safe skip distance
            move_group.set_joint_value_target(start_joint_angles)
            current_pose = move_group.get_current_pose().pose
            
            # Skip waypoints that are too close to the starting position (within 0.3m)
            # This prevents the robot from trying to move through the table surface
            skip_distance = 0.3  # meters
            filtered_waypoints = []
            
            for i, (x, y) in enumerate(sampled_waypoints):
                distance_from_start = math.sqrt((x - current_pose.position.x)**2 + 
                                              (y - current_pose.position.y)**2)
                if distance_from_start >= skip_distance or i >= len(sampled_waypoints) - 2:
                    # Keep waypoints that are far enough or the last few waypoints
                    filtered_waypoints.append((x, y))
            
            # Ensure we have at least the final waypoint
            if not filtered_waypoints and sampled_waypoints:
                filtered_waypoints = [sampled_waypoints[-1]]
            
            sampled_waypoints = filtered_waypoints
            
            rospy.loginfo("Using %d waypoints for trajectory planning (from %d path points, %d after collision filtering)", 
                         len(sampled_waypoints), len(world_path), len(filtered_waypoints))
            
            # Create waypoint poses
            waypoint_poses = []
            if sampled_waypoints:  # Only create waypoints if we have filtered waypoints
                for i, (x, y) in enumerate(sampled_waypoints[1:]):  # Skip first point (current position)
                    waypoint_pose = copy.deepcopy(final_pose)
                    waypoint_pose.position.x = x
                    waypoint_pose.position.y = y
                    waypoint_pose.position.z = self.dijkstra_height
                    waypoint_poses.append(waypoint_pose)
                
                # Set the final pose as the last waypoint
                if waypoint_poses:
                    waypoint_poses[-1] = final_pose
            
            # If no intermediate waypoints after filtering, use direct planning
            if not waypoint_poses:
                rospy.loginfo("No intermediate waypoints after filtering, using direct planning")
                return self._plan_direct_moveit(move_group, final_pose, start_joint_angles)
            
            # Plan through waypoints
            move_group.clear_pose_targets()
            move_group.set_pose_targets(waypoint_poses)
            
            # Set start state
            current_joint_state = JointState()
            current_joint_state.name = joint_names
            current_joint_state.position = start_joint_angles
            
            moveit_robot_state = RobotState()
            moveit_robot_state.joint_state = current_joint_state
            move_group.set_start_state(moveit_robot_state)
            
            # Configure planning with more tolerance for waypoint sequences
            move_group.set_planner_id(self.planner_id)
            move_group.set_planning_time(self.planning_time + 10.0)  # Extra time for complex waypoint planning
            move_group.set_num_planning_attempts(5)  # Multiple attempts
            move_group.set_goal_tolerance(0.01)  # Allow small goal tolerance 
            
            plan = move_group.plan()
            
            if plan and plan[1].joint_trajectory.points:
                rospy.loginfo("Successfully planned trajectory through waypoints")
                return planCompat(plan)
            else:
                rospy.logwarn("Waypoint planning failed, trying direct planning")
                return self._plan_direct_moveit(move_group, final_pose, start_joint_angles)
                
        except Exception as e:
            rospy.logerr("Waypoint planning failed: %s. Falling back to direct planning", e)
            return self._plan_direct_moveit(move_group, final_pose, start_joint_angles)
    
    def _plan_direct_moveit(self, move_group, destination_pose, start_joint_angles, max_attempts=None):
        """Original MoveIt! planning logic with configurable parameters"""
        if max_attempts is None:
            max_attempts = self.max_planning_attempts
            
        for attempt in range(max_attempts):
            try:
                current_joint_state = JointState()
                current_joint_state.name = joint_names
                current_joint_state.position = start_joint_angles

                moveit_robot_state = RobotState()
                moveit_robot_state.joint_state = current_joint_state
                move_group.set_start_state(moveit_robot_state)

                move_group.set_planner_id(self.planner_id)
                move_group.set_planning_time(self.planning_time)

                move_group.set_pose_target(destination_pose)
                plan = move_group.plan()

                if plan and plan[1].joint_trajectory.points:
                    return planCompat(plan)

            except Exception as e:
                rospy.logwarn("Planning attempt %d failed: %s", attempt + 1, e)

        raise Exception("Trajectory planning failed after %d attempts." % max_attempts)
    
    def plan_pick_and_place(self, req):
        """
        Enhanced pick and place planning with Dijkstra preprocessing
        Maintains exact same interface as original mover.py
        """
        response = MoverServiceResponse()

        group_name = "arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        current_robot_joint_configuration = req.joints_input.joints

        try:
            # Pre grasp - position gripper directly above target object
            rospy.loginfo("Planning pre-grasp trajectory...")
            # Adjust pose to dijkstra_height for 2D planning, then descend to actual target
            pre_grasp_planning_pose = copy.deepcopy(req.pick_pose)
            pre_grasp_planning_pose.position.z = self.dijkstra_height # Plan at safe height
            pre_grasp_pose = self._plan_trajectory_with_dijkstra(
                move_group, req.pick_pose, current_robot_joint_configuration, pre_grasp_planning_pose)
            
            if not pre_grasp_pose.joint_trajectory.points:
                rospy.logwarn("Pre-grasp planning failed")
                return response

            previous_ending_joint_angles = pre_grasp_pose.joint_trajectory.points[-1].positions

            # Grasp - lower gripper so that fingers are on either side of object (VERTICAL MOVEMENT)
            rospy.loginfo("Planning grasp trajectory...")
            pick_pose = copy.deepcopy(req.pick_pose)
            pick_pose.position.z -= 0.05
            grasp_pose = self._plan_direct_moveit(
                move_group, pick_pose, previous_ending_joint_angles)
            
            if not grasp_pose.joint_trajectory.points:
                rospy.logwarn("Grasp planning failed")
                return response

            previous_ending_joint_angles = grasp_pose.joint_trajectory.points[-1].positions

            # Pick Up - raise gripper back to the pre grasp position (VERTICAL MOVEMENT)
            rospy.loginfo("Planning pick-up trajectory...")
            pick_up_pose = self._plan_direct_moveit(
                move_group, req.pick_pose, previous_ending_joint_angles)
            
            if not pick_up_pose.joint_trajectory.points:
                rospy.logwarn("Pick-up planning failed")
                return response

            previous_ending_joint_angles = pick_up_pose.joint_trajectory.points[-1].positions

            # Place - move gripper to desired placement position
            rospy.loginfo("Planning place trajectory...")
            # First plan 2D transport at safe height, final waypoint will use precise Z
            place_planning_pose = copy.deepcopy(req.place_pose)
            place_planning_pose.position.z = self.dijkstra_height  # Plan at safe height
            place_pose = self._plan_trajectory_with_dijkstra(
                move_group, req.place_pose, previous_ending_joint_angles, place_planning_pose)

            if not place_pose.joint_trajectory.points:
                rospy.logwarn("Place planning failed")
                return response

            # Success - add all trajectories to response
            response.trajectories.append(pre_grasp_pose)
            response.trajectories.append(grasp_pose)
            response.trajectories.append(pick_up_pose)
            response.trajectories.append(place_pose)
            
            rospy.loginfo("Successfully planned complete pick-and-place sequence")

        except Exception as e:
            rospy.logerr("Pick and place planning failed: %s", e)
            return response
        
        finally:
            move_group.clear_pose_targets()

        return response


def dijkstra_moveit_server():
    """
    Initialize the enhanced Dijkstra moveit server and set up the collision object subscriber.
    """
    try:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('ur10e_rg2_moveit_server')
        
        # Create the enhanced server
        server = DijkstraMoveitServer()
        
        # Set up the service for trajectory planning (same interface as original)
        s = rospy.Service('ur10e_rg2_moveit', MoverService, server.plan_pick_and_place)
        rospy.loginfo("Ready to plan!")

        # Set up the subscriber for collision objects
        rospy.Subscriber("/collision_object", CollisionObject, server.collision_callback)
        
        # Initialize grid with workspace boundaries
        server._update_grid()
        
        rospy.spin()
        
    except Exception as e:
        rospy.logerr("Failed to start Dijkstra Moveit Server: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    dijkstra_moveit_server()