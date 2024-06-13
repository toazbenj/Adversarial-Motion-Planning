#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
import scipy.linalg as sp
import os
import pdb
import pickle
import time
import argparse


class LQR():
    def __init__(self,ns):
        # Function to perform when shut down.
        rospy.on_shutdown(self.shutdown)
        # Namespace.
        self.ns = ns
        
        
    def test(self,data):
        print(data)
        
        
    def control(self):
#        print(data)
    
        # Create a publisher.
        self.r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 1)
        
        # Blank message.
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        
        vel_msg.angular.z = 0.17 
        vel_msg.linear.x = 0.18
        # Publish the message.
        self.r_vel.publish(vel_msg)
       
        
    def shutdown(self):
        print('Begin shutdown')
        self.r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 1)
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0
        self.r_vel.publish(vel_msg)
        # Log the data.
        rospy.loginfo("Stop")
        rospy.sleep(1)

parser = argparse.ArgumentParser(description = "Rotary Control")
parser.add_argument("--ns", type=str, default=None, help="Robot Name")
        
def main(ns):

    robot_game = LQR(ns)
    r = rospy.Rate(20)
    while not rospy.is_shutdown():
        robot_game.control()
        r.sleep()

if __name__ == '__main__':
    print('Start')
    # Initialize the node.
    rospy.init_node('rotary_traj', anonymous = True, disable_signals=True)
    # Read the parser.
    args= parser.parse_args()
    main(args.ns)
    