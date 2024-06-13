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
#os.environ["ROS_NAMESPACE"] = "Robot1"


class LQR():
    def __init__(self,ns):
        # Function to perform when shut down.
        rospy.on_shutdown(self.shutdown)
        # Namespace.
        self.ns = ns
        # Time step.
        self.dt = 0.2
        
        # Discrete time dynamics.
        self.A = [[1.0,0.0],[0.0,1.0]]
        self.B = [[self.dt,0.0],[0.0,self.dt]]
#        self.A = [[1.0]]
#        self.B = [[self.dt]]
        self.A = np.asmatrix(self.A)
        self.B = np.asmatrix(self.B)
        
        # State and control weights.
        self.Q = np.eye((2))
        self.R = np.eye((2))
        
        # Solve the LQR gain.
        self.K = self.dare()
#        print(self.K)
#        pdb.set_trace()
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        # Save the data.
        self.f = open(cur_dir + "/dummy.pkl","wb")
        # Theta tracking.
        self.theta = []
        # Theta tracking reference.
        self.theta_r = []
        # Time store.
        self.time = []
        t = rospy.Time.from_sec(time.time())
        self.init_time = t.to_sec()
        self.time.append(self.init_time)
#        pickle.dump(10,self.f)
#        f.close()
        # Subscribe.
        self.sub = rospy.Subscriber("/mocap_node/Robot_2/Odom", Odometry, self.control)
#        self.sub = rospy.Subscriber("/Robot1/odom", Odometry, self.test)
        # Proportional gain.
        self.Kp = 10.0
        # Integral gain.
        self.Ki = 0.0
        # Differential gain.
        self.Kd = 5.0
        # Integral error.
        self.i_e = 0.0
        # Safe angular output.
        self.ang_max = 1.5
        # Derivative error.
        self.dr = 0.0
        # Rate of loop.
        self.rate = 10.0
        # Save position data.
        self.pos_data = []
        self.l_nv = 1.0
        
    def test(self,data):
        print(data)
        
    def dare(self):
        '''
        Solve the LQR problem.
        '''
        # Solve ricatti equation.
        X = np.matrix(sp.solve_discrete_are(self.A, self.B, self.Q, self.R))
        # Compute the LQR gain.
        K = -np.matrix(sp.inv(self.B.T*X*self.B + self.R)*(self.B.T*X*self.A))
        
        return K
    
    def pid(self, theta_r, l_nv, yaw):
        # Create a publisher.
        self.r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 1)
        
        # Blank message.
        vel_msg = Twist()
        vel_msg.linear.x = l_nv
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        # Calculate the error.
        err_theta = theta_r - yaw
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
        vel_msg.angular.z = Kp + Ki + Kd
        # Publish the message.
        self.r_vel.publish(vel_msg)
    
    def control(self,data):
        '''
        Realize LQR control.
        '''
        
        # Angular data.
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.theta.append(yaw)
        # Determine the position.
        pos = data.pose.pose.position
        # Linear velocity gain. 
        l_v = np.matmul(-self.K,(np.array([[pos.x],[pos.y]])))
        
        
#        pdb.set_trace()
        # Determine magnitude of velocity.
        l_nv = np.linalg.norm(l_v)
        self.l_nv = l_nv
#        pdb.set_trace()
        # Determine angle.
        an = np.arctan2(l_v[1],l_v[0])
        # Store the reference theta.
        self.theta_r.append(an[0,0])
#        print(an[0,0])
        # Call PID control.
        self.pid(an[0,0], l_nv, yaw)
        # Save position data.
        self.pos_data.append(np.array([[pos.x],[pos.y]]))
        print(l_nv)
        if l_nv <= 0.05:
            rospy.signal_shutdown("Reactor shutting down.")
        
        
        
        
    def tune(self,data):
#        print(data)
    
        # Create a publisher.
        self.r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 1)
        
        # Tune the PID controller.
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        # Store the theta
        self.theta.append(yaw)
        # Append time.
        t = rospy.Time.from_sec(time.time())
        self.time.append(t.to_sec())
        if self.time[-1] - self.init_time >= 8.0:
            print('Start tracking')
            r_theta = 1.00
        else:
            r_theta = 0.00
#        print((self.time[-1] - self.init_time))
#        r_theta = 0.0
        # Store the reference theta.
        self.theta_r.append(r_theta)
        # Blank message.
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        # Calculate the error.
        err_theta = r_theta - yaw
#        print(err_theta)
        # Proportional gain.
        Kp = self.Kp*err_theta
        # Integral gain.
        Ki = self.Ki*(err_theta + self.i_e)
        # Update the integral term.
        self.i_e = self.i_e + err_theta
        # Anti windup mechanism.
        if np.abs(Ki) >= self.ang_max:
            Ki = np.sign(Ki)*self.ang_max
        print(Ki)
        # Derivative gain.
        Kd = self.Kd*(err_theta - self.dr)/self.rate
        self.dr = err_theta
        vel_msg.angular.z = Kp + Ki + Kd
        vel_msg.angular.z = 0.0
        vel_msg.linear.x = 1.0
        print(vel_msg)
        # Publish the message.
        self.r_vel.publish(vel_msg)
       
        
    def shutdown(self):
        print('Begin shutdown')
        # Save the data.
        with open('position_ver2.npy', 'wb') as f:
            np.save(f, np.array(self.pos_data))
        self.r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 1)
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0
        self.r_vel.publish(vel_msg)
        # Save the data.
        with open('theta_LQR_r1.npy', 'wb') as f:
            np.save(f, np.array(self.theta))
        # Save the data.
        with open('theta_ref_LQR_r1.npy', 'wb') as f:
            np.save(f, np.array(self.theta_r))
        # Log the data.
        rospy.loginfo("Stop")
        rospy.sleep(1)

#
#def callback(data):
#    # Read the theta.
##    print(data.pose.pose.orientation)
#    orientation_q = data.pose.pose.orientation
#    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
#    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
#    # Reference theta.
#    r_theta = 3.14
#    
#    # Publish to 
#    r_vel = rospy.Publisher("/cmd_vel", Twist, queue_size = 2)
#    
#    # Blank message.
#    vel_msg = Twist()
#    vel_msg.linear.x = 0.0
#    vel_msg.linear.y = 0.0
#    vel_msg.linear.z = 0.0
#    
#    vel_msg.angular.x = 0.0
#    vel_msg.angular.y = 0.0
#    # Calculate the error.
#    err_theta = r_theta - yaw
#    vel_msg.angular.z = err_theta
#    # Publish the message.
#    r_vel.publish(vel_msg)
#    
##    return None
#    
#def cmd_vel(data):
#    print(data.angular)
#    
#def listener():
#
#    # In ROS, nodes are uniquely named. If two nodes with the same
#    # name are launched, the previous one is kicked off. The
#    # anonymous=True flag means that rospy will choose a unique
#    # name for our 'listener' node so that multiple listeners can
#    # run simultaneously.
#    rospy.init_node('optitrack_control', anonymous=True)
#    print('Ok')
#    rospy.Subscriber("/mocap_node/Robot_test_1/Odom", Odometry, callback)
#    
#    # Publish to 
##    r_vel = rospy.Publisher("/cmd_vel", Twist)
##    print(r_vel.angular)
#
#    # spin() simply keeps python from exiting until this node is stopped
#    rospy.spin()
        
parser = argparse.ArgumentParser(description = "LQR Control")
parser.add_argument("--ns", type=str, default=None, help="Robot Name")
        
def main(ns):

    robot_game = LQR(ns)
    r = rospy.Rate(20)
    while not rospy.is_shutdown():
        r.sleep()
#        if robot_game.l_nv <= 0.1:
#            rospy.signal_shutdown("Reactor shutting down.")
#            rospy.spin()
#    try:
#        while not rospy.is_shutdown():
#            rospy.spin()
#    except KeyboardInterrupt:
#        print("Shutting down")

if __name__ == '__main__':
    print('Start')
    # Initialize the node.
    rospy.init_node('optitrack_control', anonymous = True, disable_signals=True)
    # The dynamic game.
    # Read the parser.
    args= parser.parse_args()
    print(args.ns)
    main(args.ns)
    
#    try:
#        robot_game.control()    
#    except rospy.ROSInterruptException:
#        pass