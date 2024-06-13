#!/usr/bin/env python 
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion
import pdb
import argparse

class GoToPose():
    def __init__(self):

        self.goal_sent = False

	# What to do if shut down (e.g. Ctrl-C or failure)
	rospy.on_shutdown(self.shutdown)
	
	# Tell the action client that we want to spin a thread by default
	self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
	rospy.loginfo("Wait for the action server to come up")

	# Allow up to 5 seconds for the action server to come up
	self.move_base.wait_for_server(rospy.Duration(5))

    def goto(self, pos, quat):

        # Send a goal
        self.goal_sent = True
	goal = MoveBaseGoal()
	goal.target_pose.header.frame_id = 'map'
	goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = Pose(Point(pos['x'], pos['y'], 0.000),
                                     Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4']))

	# Start moving
        self.move_base.send_goal(goal)

	# Allow TurtleBot up to 60 seconds to complete task
	success = self.move_base.wait_for_result() 

        state = self.move_base.get_state()
        result = False

        if success and state == GoalStatus.SUCCEEDED:
            # We made it!
            result = True
        else:
            self.move_base.cancel_goal()

        self.goal_sent = False
        return result

    def shutdown(self):
        if self.goal_sent:
            self.move_base.cancel_goal()
        rospy.loginfo("Stop")
        rospy.sleep(1)


# Create parser (for user input)
parser = argparse.ArgumentParser(description = "Navigation")
parser.add_argument("--X", type=int, default=0.0, help="X position")
parser.add_argument("--Y", type=int, default=1.25, help="Y position")
parser.add_argument("--r1", type=int, default=0.000, help="R1 quaternion")
parser.add_argument("--r2", type=int, default=0.000, help="R2 quaternion")
parser.add_argument("--r3", type=int, default=0.707, help="R3 quaternion")
parser.add_argument("--r4", type=int, default=0.707, help="R4 quaternion")


if __name__ == '__main__':
    try:
        rospy.init_node('nav_test', anonymous=False)
        navigator = GoToPose()

        # Read the parser.
        args=parser.parse_args()

        # Customize the following values so they are appropriate for your location
        #position = {'x': 1.22, 'y' : 2.56}
        #quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : 0.000, 'r4' : 1.000}

        # Desored position of the robot.
        position_x = args.X
        position_y = args.Y

        # Desired orientation of the robot.
        rot_r1 = args.r1
        rot_r2 = args.r2
        rot_r3 = args.r3
        rot_r4 = args.r4

        # Conver to dict for passing into ROS.
        position = {'x': position_x, 'y' : position_y}
        quaternion = {'r1' : rot_r1, 'r2' : rot_r2, 'r3' : rot_r3, 'r4' : rot_r4}        

        rospy.loginfo("Go to (%s, %s) pose", position['x'], position['y'])
        success = navigator.goto(position, quaternion)

        if success:
            rospy.loginfo("Desired goal reached")
        else:
            rospy.loginfo("The base failed to reach the desired goal")

        # Sleep to give the last log messages time to be sent
        rospy.sleep(1)

    except rospy.ROSInterruptException:
        rospy.loginfo("Ctrl-C caught. Quitting")