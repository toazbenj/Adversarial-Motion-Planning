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
        # Subscriber.
        self.sub = rospy.Subscriber("/mocap_node/" + self.ns + "/Odom", Odometry, self.control)
        # Proportional gain.
        self.Kp = 0.5
        # Derivative gain.
        self.Kd = 0.2
        # Integral gain.
        self.Ki = 0.0
        # Integral error.
        self.i_e = 0.0
        # Safe angular output.
        self.ang_max = 1.5
        # Derivative error.
        self.dr = 0.0
        # Rate of loop.
        self.rate = 20.0
        # Reference radius.
        self.r_ref = 1.0
        rospy.on_shutdown(self.pre_shut)
        
    def test(self,data):
        print(data)
        
    def control(self, data):
        # Get the position data.
        pos = data.pose.pose.position
        # Determine the angle.
        r_theta = np.arctan2(pos.y, pos.x)
        
        # Determine the radius of the robot. 
        r_robot = np.linalg.norm(np.array([pos.x,pos.y]))
        
        # Create a publisher.
        self.r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 1)
        
        # Blank message.
        vel_msg = Twist()
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
         
        vel_x_r = 0.16
        # Set linear speed.
        vel_msg.linear.x = vel_x_r
        
        # Calculate the error.
        err_temp = r_robot - self.r_ref
        if err_temp >= 0.05:
            err_theta = err_temp
        else:
            err_theta = 0.0
        # Proportional gain.
        Kp = self.Kp*err_theta
        # Integral gain.
        Ki = self.Ki*(err_theta + self.i_e)
        # Update the integral term.
        self.i_e = self.i_e + err_theta
        # Anti windup mechanism.
        if np.abs(Ki) >= self.ang_max:
            Ki = np.sign(Ki)*self.ang_max
            Ki = 0.1
        # Derivative gain.
        Kd = self.Kd*(err_theta - self.dr)/self.rate
        self.dr = err_theta
        print('PID Error')
        print(Kp + Ki + Kd)
        print('Angular error')
        print(err_theta)
        print(Ki)
        vel_msg.angular.z = vel_x_r + Kp + Ki + Kd
        # Publish the message.
        self.r_vel.publish(vel_msg)
       
    def pre_shut(self):
        r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 1)
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0
        r_vel.publish(vel_msg)
        
    def shutdown(self):
        print('Begin shutdown')
        r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 10)
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0
        r_vel.publish(vel_msg)
        # Log the data.
        rospy.loginfo("Stop")
        rospy.sleep(1.0)

parser = argparse.ArgumentParser(description = "Rotary Control")
parser.add_argument("--ns", type=str, default=None, help="Robot Name")
        
def main(ns):

    robot_game = LQR(ns)
    r = rospy.Rate(20)
    while not rospy.is_shutdown():
        r.sleep()

if __name__ == '__main__':
    print('Start')
    # Initialize the node.
    rospy.init_node('rotary_traj', anonymous = True, disable_signals=True)
    # Read the parser.
    args= parser.parse_args()
    main(args.ns)
    