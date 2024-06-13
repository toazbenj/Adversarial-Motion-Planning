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
        # Namespace.
        self.ns = ns
        # Initialize the node.
        rospy.init_node('straight_traj', anonymous = True, disable_signals=True)
        # Function to perform when shut down.
        rospy.on_shutdown(self.shutdown)
        # Create a publisher.
        self.r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist,queue_size = 1)
        # Counter.
        self.cnt = 0
        self.data = Odometry()
        # Initialize a subscriber.
        rospy.wait_for_message("/mocap_node/" +  self.ns + "/Odom", Odometry,2.0)
        rospy.Subscriber("/mocap_node/" +  self.ns + "/Odom", Odometry, self.test)
        
        
    
        
    def test(self,data):
#        print(data)
        self.data = data
        

        
        
    def main_loop(self):
        print('Start')
        
        # Empty message.
        self.data = Odometry()
        
        rate = rospy.Rate(1.0)
        
        while not rospy.is_shutdown():
            
            # Subscribe to current robot.
            try:
                rospy.Subscriber("/mocap_node/" +  self.ns + "/Odom", Odometry, self.test)
            except rospy.ROSInterruptException:
                pass
                
#            print(self.data)
            
            # Blank message.
            vel_msg = Twist()
            vel_msg.linear.x = 0.0
            vel_msg.linear.y = 0.0
            vel_msg.linear.z = 0.0
            
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            
            vel_msg.angular.z = 0.0 
            vel_msg.linear.x = 0.14
            
            self.r_vel.publish(vel_msg)
            rate.sleep()
            
            self.cnt = self.cnt + 1
            
            if self.cnt >= 25:
                rospy.signal_shutdown("Reactor shutting down.")
        
        
        
    def control(self,data):
        # Loop frequency.
        
        print(self.cnt)
        # Track the position.
        pos_self = data.pose.pose.position
        
        
        
        
        
#        print(data)
        
#        if pos_self.x <= 2.0:
#            # Publish the message.
#            self.r_vel.publish(vel_msg)
#        else:
#            print(self.cnt)
#            rospy.signal_shutdown("Reactor shutting down.")
        
    def shutdown(self):
        print('Begin shutdown')
#        r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 1,latch=True)
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

if __name__ == '__main__':
    print('Start')
    
    # Read the parser.
    args= parser.parse_args()
    
    robot_game = LQR(args.ns)
    
    try:
        robot_game.main_loop()    
    except rospy.ROSInterruptException:
        pass
    
