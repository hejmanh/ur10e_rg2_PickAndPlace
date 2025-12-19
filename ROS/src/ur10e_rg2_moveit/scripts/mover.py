#!/usr/bin/env python

from __future__ import print_function

import rospy
import sys
import copy
import math
import moveit_commander

import moveit_msgs.msg
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, BoundingVolume
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, CollisionObject
from shape_msgs.msg import SolidPrimitive
import geometry_msgs.msg
from geometry_msgs.msg import Quaternion, Pose
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from ur10e_rg2_moveit.srv import MoverService, MoverServiceRequest, MoverServiceResponse

joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

# Between Melodic and Noetic, the return type of plan() changed. moveit_commander has no __version__ variable, so checking the python version as a proxy
if sys.version_info >= (3, 0):
    def planCompat(plan):
        return plan[1]
else:
    def planCompat(plan):
        return plan

"""
    Callback function to integrate collision objects with MoveIt's planning scene.
    This allows the RRT planner to avoid dynamic obstacles from Unity.
"""
def collision_callback(msg):
    rospy.loginfo("Received collision object: %s (operation: %d)", msg.id, msg.operation)
    
    try:
        # Get the planning scene interface (use asynchronous mode for collision object publisher)
        scene = moveit_commander.PlanningSceneInterface(synchronous=False)
        
        if msg.operation == msg.ADD:
            rospy.loginfo("Adding collision object '%s' to planning scene", msg.id)
            
            if len(msg.mesh_poses) > 0 and len(msg.meshes) > 0:
                # Log the pose information
                pose = msg.mesh_poses[0]
                rospy.loginfo("Pose: position(%.3f, %.3f, %.3f) frame: %s", 
                            pose.position.x, pose.position.y, pose.position.z, msg.header.frame_id)
                
                # Use the add_object method to add the collision object directly
                scene.add_object(msg)
                rospy.loginfo("Successfully added mesh collision object '%s' to planning scene", msg.id)
                
            # elif len(msg.primitive_poses) > 0 and len(msg.primitives) > 0:
            #     # Handle primitive shapes (box, sphere, cylinder)
            #     pose = msg.primitive_poses[0]
            #     primitive = msg.primitives[0]
            #     rospy.loginfo("Pose: position(%.3f, %.3f, %.3f) frame: %s", 
            #                 pose.position.x, pose.position.y, pose.position.z, msg.header.frame_id)
                
            #     # Create a proper PoseStamped for primitives
            #     from geometry_msgs.msg import PoseStamped
            #     pose_stamped = PoseStamped()
            #     pose_stamped.header.frame_id = msg.header.frame_id
            #     pose_stamped.pose = pose
                
            #     if primitive.type == primitive.BOX:
            #         # primitive.dimensions[0] = x, [1] = y, [2] = z
            #         size = (primitive.dimensions[0], primitive.dimensions[1], primitive.dimensions[2])
            #         scene.add_box(msg.id, pose_stamped, size)
            #     elif primitive.type == primitive.SPHERE:
            #         # primitive.dimensions[0] = radius
            #         radius = primitive.dimensions[0]
            #         scene.add_sphere(msg.id, pose_stamped, radius)
            #     elif primitive.type == primitive.CYLINDER:
            #         # primitive.dimensions[0] = height, [1] = radius
            #         height = primitive.dimensions[0]
            #         radius = primitive.dimensions[1]
            #         scene.add_cylinder(msg.id, pose_stamped, height, radius)
            #     else:
            #         rospy.logwarn("Unsupported primitive type: %d for collision object '%s'", primitive.type, msg.id)
            #         return
                    
            #     rospy.loginfo("Successfully added primitive collision object '%s' to planning scene", msg.id)
                
            else:
                rospy.logwarn("No mesh or primitive data found for collision object '%s'", msg.id)
                
        elif msg.operation == msg.REMOVE:
            rospy.loginfo("Removing collision object '%s' from planning scene", msg.id)
            scene.remove_world_object(msg.id)
            
        elif msg.operation == msg.MOVE:
            rospy.loginfo("Moving collision object '%s'", msg.id)
            # For move operations, remove the old object and add the new one
            scene.remove_world_object(msg.id)
            rospy.sleep(0.1)  # Allow time for removal to propagate
            
            # Re-add the object with updated pose
            if len(msg.mesh_poses) > 0 and len(msg.meshes) > 0:
                # For mesh objects, use add_object
                scene.add_object(msg)
            elif len(msg.primitive_poses) > 0 and len(msg.primitives) > 0:
                pose = msg.primitive_poses[0]
                primitive = msg.primitives[0]
                
                # Create a proper PoseStamped for primitives
                from geometry_msgs.msg import PoseStamped
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = msg.header.frame_id
                pose_stamped.pose = pose
                
                if primitive.type == primitive.BOX:
                    size = (primitive.dimensions[0], primitive.dimensions[1], primitive.dimensions[2])
                    scene.add_box(msg.id, pose_stamped, size)
                elif primitive.type == primitive.SPHERE:
                    radius = primitive.dimensions[0]
                    scene.add_sphere(msg.id, pose_stamped, radius)
                elif primitive.type == primitive.CYLINDER:
                    height = primitive.dimensions[0]
                    radius = primitive.dimensions[1]
                    scene.add_cylinder(msg.id, pose_stamped, height, radius)
            
            rospy.loginfo("Successfully moved collision object '%s'", msg.id)
            
        else:
            rospy.logwarn("Unknown collision object operation: %d", msg.operation)
            
        # Allow time for planning scene update to propagate
        rospy.sleep(0.1)
        
        # Verify the object was added/removed
        known_objects = scene.get_known_object_names()
        if msg.operation == msg.ADD and msg.id in known_objects:
            rospy.loginfo("Confirmed: '%s' is now in planning scene", msg.id)
        elif msg.operation == msg.REMOVE and msg.id not in known_objects:
            rospy.loginfo("Confirmed: '%s' removed from planning scene", msg.id)
        
    except Exception as e:
        rospy.logerr("Failed to process collision object '%s': %s", msg.id, str(e))


def plan_trajectory(move_group, destination_pose, start_joint_angles, max_attempts=100): 
    for attempt in range(max_attempts):
        try:
            current_joint_state = JointState()
            current_joint_state.name = joint_names
            current_joint_state.position = start_joint_angles

            moveit_robot_state = RobotState()
            moveit_robot_state.joint_state = current_joint_state
            move_group.set_start_state(moveit_robot_state)

            move_group.set_planner_id("RRT")
            move_group.set_planning_time(15.0)

            move_group.set_pose_target(destination_pose)
            plan = move_group.plan()

            if plan and plan[1].joint_trajectory.points:
                return planCompat(plan)

        except Exception as e:
            rospy.logwarn(f"Planning attempt {attempt + 1} failed: {e}")

    raise Exception(f"Trajectory planning failed after {max_attempts} attempts.")
    
"""
    Creates a pick and place plan using the four states below.
    
    1. Pre Grasp - position gripper directly above target object
    2. Grasp - lower gripper so that fingers are on either side of object
    3. Pick Up - raise gripper back to the pre grasp position
    4. Place - move gripper to desired placement position

    Gripper behaviour is handled outside of this trajectory planning.
        - Gripper close occurs after 'grasp' position has been achieved
        - Gripper open occurs after 'place' position has been achieved

    https://github.com/ros-planning/moveit/blob/master/moveit_commander/src/moveit_commander/move_group.py
"""
def plan_pick_and_place(req):
    response = MoverServiceResponse()

    group_name = "arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    current_robot_joint_configuration = req.joints_input.joints

    # Pre grasp - position gripper directly above target object
    pre_grasp_pose = plan_trajectory(move_group, req.pick_pose, current_robot_joint_configuration)
    
    # If the trajectory has no points, planning has failed and we return an empty response
    if not pre_grasp_pose.joint_trajectory.points:
        return response

    previous_ending_joint_angles = pre_grasp_pose.joint_trajectory.points[-1].positions

    # Grasp - lower gripper so that fingers are on either side of object
    pick_pose = copy.deepcopy(req.pick_pose)
    pick_pose.position.z -= 0.05  # Static value coming from Unity, TODO: pass along with request
    grasp_pose = plan_trajectory(move_group, pick_pose, previous_ending_joint_angles)
    
    if not grasp_pose.joint_trajectory.points:
        return response

    previous_ending_joint_angles = grasp_pose.joint_trajectory.points[-1].positions

    # Pick Up - raise gripper back to the pre grasp position
    pick_up_pose = plan_trajectory(move_group, req.pick_pose, previous_ending_joint_angles)
    
    if not pick_up_pose.joint_trajectory.points:
        return response

    previous_ending_joint_angles = pick_up_pose.joint_trajectory.points[-1].positions

    # Place - move gripper to desired placement position
    place_pose = plan_trajectory(move_group, req.place_pose, previous_ending_joint_angles)

    if not place_pose.joint_trajectory.points:
        return response

    # If trajectory planning worked for all pick and place stages, add plan to response
    response.trajectories.append(pre_grasp_pose)
    response.trajectories.append(grasp_pose)
    response.trajectories.append(pick_up_pose)
    response.trajectories.append(place_pose)

    move_group.clear_pose_targets()

    return response

def moveit_server():
    """
    Initialize the moveit server and set up the collision object subscriber.
    """
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('ur10e_rg2_moveit_server')

    # Set up the service for trajectory planning
    s = rospy.Service('ur10e_rg2_moveit', MoverService, plan_pick_and_place)
    print("Ready to plan")

    # Set up the subscriber for collision objects
    rospy.Subscriber("/collision_object", CollisionObject, collision_callback)

    rospy.spin()


if __name__ == "__main__":
    moveit_server()
