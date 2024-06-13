#!/usr/bin/env python
import rospy
import message_filters
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


class Follower():
    def __init__(self,args):
        # Function to perform when shut down.
        rospy.on_shutdown(self.shutdown)
        # Namespace.
        self.args = args
        # Proportional gain.
        self.Kp = 0.5
        # Derivative gain.
        self.Kd = 0.1
        # Integral gain.
        self.Ki = 0.00
        # Integral error.
        self.i_e = 0.0
        # Safe angular output.
        self.ang_max = 1.5
        # Derivative error.
        self.dr = 0.0
        # Rate of loop.
        self.rate = 10.0
        # Reference radius.
        self.r_ref = 1.0
        # Subscribe to the agent ahead.
        self.agent_a = rospy.Subscriber("/mocap_node/"+ str(self.args.r_a)+ "/Odom", Odometry, self.dummy)
        # Subscribe to current robot.
        self.agent_self = rospy.Subscriber("/mocap_node/"+ str(self.args.r_f)+ "/Odom", Odometry, self.control)
        
    def dummy(self,data):
        self.data_a = data
        
    def test(self,data):
        print(data)
        
    def control(self, agent_self):
#        print(self.data_a.pose)
        # Position of the robot ahead.
        pos_ahead = self.data_a.pose.pose.position
        # Position of self.
        pos_self = agent_self.pose.pose.position
        # Determine the radius of the robot. 
        r_robot = np.linalg.norm(np.array([pos_self.x,pos_self.y]))
        
        
#        print(pos_self)
        # Calculate the arc between robots.
        theta_ahead = np.arctan2(pos_ahead.y,pos_ahead.x)
        theta_follow = np.arctan2(pos_self.x,pos_self.y)
        
        d_theta = np.mod(theta_ahead,2*np.pi) - np.mod(theta_follow,2*np.pi)
        print(d_theta)
        
        # Determine policy. 
        if d_theta <= 0.95*np.pi/2:
            vel_l = 0.16
            vel_a = 0.16
        else:
            vel_l = 0.18
            vel_a = 0.18
        vel_l = 0.18
        vel_a = 0.18
        # PID.
        # Calculate the error.
        err_theta = r_robot - self.r_ref
        # Proportional gain.
        Kp = self.Kp*err_theta
        # Integral gain.
        Ki = self.Ki*(err_theta + self.i_e)
        # Update the integral term.
        self.i_e = self.i_e + err_theta
        # Anti windup mechanism.
        if np.abs(Ki) >= self.ang_max:
            Ki = np.sign(Ki)*self.ang_max
        # Derivative gain.
        Kd = self.Kd*(err_theta - self.dr)/self.rate
        self.dr = err_theta
        
        # Create a publisher.
        self.r_vel = rospy.Publisher("/" + self.args.r_f + "/cmd_vel", Twist, queue_size = 1)
        
        # Blank message.
        vel_msg = Twist()
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        
        vel_msg.angular.z = vel_a + Kp + Ki + Kd
        vel_msg.linear.x = vel_l
        # Publish the message.
        self.r_vel.publish(vel_msg)
       
        
    def shutdown(self):
        print('Begin shutdown')
        r_vel = rospy.Publisher("/" + self.args.r_f + "/cmd_vel", Twist, queue_size = 1)
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0
        r_vel.publish(vel_msg)

parser = argparse.ArgumentParser(description = "Rotary Control follower")
parser.add_argument("--r_f", type=str, default=None, help="Robot following")
parser.add_argument("--r_a", type=str, default=None, help="Robot ahead")
        
def main(args):

    robot_game = Follower(args)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        r.sleep()

if __name__ == '__main__':
    print('Start')
    # Initialize the node.
    rospy.init_node('rotary_traj_follower', anonymous = True, disable_signals=True)
    # Read the parser.
    args= parser.parse_args()
    main(args)
    