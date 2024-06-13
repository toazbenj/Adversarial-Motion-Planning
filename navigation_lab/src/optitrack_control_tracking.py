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
    def __init__(self,args):
        # Function to perform when shut down.
        rospy.on_shutdown(self.shutdown)
        # Namespace.
        self.ns = args.ns
        # X location.
        self.x_loc = args.x_loc
        # Y location.
        self.y_loc = args.y_loc
        # Theta reference.
        self.tht = args.tht
        # Bool for theta.
        if np.copy(self.tht) is not None:
            self.th_bool = True
        else:
            self.th_bool = False
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
        temp = "/mocap_node/" + self.ns + "/Odom"
        print(temp)
        # Proportional gain.
#        self.Kp = 10.0/2.0
        self.Kp = 10.0/2.0
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
        # Orientation.
        self.ort = False
        # Sequence flag.
        self.sq_fl = False
#        self.sub = rospy.Subscriber("/mocap_node/Robot_2/Odom", Odometry, self.control)
        self.sub = rospy.Subscriber("/mocap_node/" + self.ns + "/Odom", Odometry, self.control)
#        self.sub = rospy.Subscriber("/Robot1/odom", Odometry, self.test)
        rospy.spin()
        
        
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
        if np.mod(yaw,2*np.pi) >= np.mod(theta_r+ np.pi,2*np.pi) :
            sig = -1
        else:
            sig = 1
        err_theta = (theta_r - yaw)
#        err_theta = sig*np.mod(theta_r - yaw,np.pi)
        # Proportional gain.
        Kp = self.Kp*err_theta
        # Integral gain.
        Ki = self.Ki*(err_theta + self.i_e)
        if Ki != 0.0:
            # Update the integral term.
            self.i_e = self.i_e + err_theta
        # Anti windup mechanism.
        if np.abs(Ki) >= self.ang_max:
            Ki = np.sign(Ki)*self.ang_max
        # Derivative gain.
        Kd = self.Kd*(err_theta - self.dr)/self.rate
        self.dr = err_theta
        vel_msg.angular.z = Kp + Ki + Kd
#        print(vel_msg.angular.z)
#        if np.abs(vel_msg.angular.z)  >= 0.22:
#            vel_msg.angular.z = np.sign(vel_msg.angular.z)*0.22
#        print(vel_msg.angular.z)
#        print(theta_r)
#        print(yaw)
        
        # Publish the message.
        self.r_vel.publish(vel_msg)
    
    def control(self,data):
        '''
        Realize LQR control.
        '''
        
            
#        print(self.tht)
        if self.sq_fl == False:
            # Angular data.
            orientation_q = data.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
            self.theta.append(yaw)
            # Determine the position.
            pos = data.pose.pose.position
            self.data = pos
#            print(self.K)
            # Linear velocity gain. 
            l_v = np.matmul(self.K,(np.array([[pos.x - self.x_loc],[pos.y-self.y_loc]])))
#            print(l_v)
            # Determine magnitude of velocity.
            l_nv = np.linalg.norm(l_v)
            self.l_nv = l_nv
            # Determine angle.
            an = np.arctan2(l_v[1],l_v[0])
            
            # Store the reference theta.
            self.theta_r.append(an[0,0])
    #        print(an[0,0])
            # Call PID control.
            self.pid(an[0,0], l_nv, yaw)
            # Save position data.
            self.pos_data.append(np.array([[pos.x],[pos.y]]))
        if self.l_nv <= 0.05:
#            print(self.tht)
            if self.th_bool:
                # Derivative term to zero.
                self.Kd = 0.1
                # Lower the proportional term.
                self.Kp = 0.2
                self.Ki = 0.005
                if self.sq_fl == False:
                    print('Location sequence complete')
    
                self.sq_fl = True
                # Create a second sequence of aligning the angle.
                # Reference for theta.
                theta_ref = self.tht
                # Angular data.
                orientation_q = data.pose.pose.orientation
                orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    #            print(yaw)
                # Call PID control.
                self.pid(theta_ref, 0.0, yaw)
                
                if np.abs(yaw - theta_ref) <= 0.01:
                    print(yaw - theta_ref)
                    rospy.signal_shutdown("Reactor shutting down.")
            else:
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
       
    def pre_shut(self):
        self.r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 1)
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0
        
    def shutdown(self):
        print('Begin shutdown')
        # Save the data.
#        with open('position_ver2.npy', 'wb') as f:
#            np.save(f, np.array(self.pos_data))
        self.r_vel = rospy.Publisher("/" + self.ns + "/cmd_vel", Twist, queue_size = 1)
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0
        try:
            self.r_vel.publish(vel_msg)
        except rospy.ROSInterruptException:
            pass
        '''
        # Save the data.
        with open('theta_LQR_r1.npy', 'wb') as f:
            np.save(f, np.array(self.theta))
        # Save the data.
        with open('theta_ref_LQR_r1.npy', 'wb') as f:
            np.save(f, np.array(self.theta_r))
        '''
        # Log the data.
        rospy.loginfo("Stop")
        rospy.sleep(1)

        
parser = argparse.ArgumentParser(description = "LQR Control")
parser.add_argument("--ns", type=str, default=None, help="Robot Name")
parser.add_argument("--x_loc", type=float, default=0.0, help="Robot x-location")
parser.add_argument("--y_loc", type=float, default=0.0, help="Robot y-location")
parser.add_argument("--tht", type=float, default=0.0, help="Robot orientation")
        
#def main(args):
#
#    robot_game = LQR(args)
#    r = rospy.Rate(10)
#    while not rospy.is_shutdown():
#        r.sleep()
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
    rospy.init_node('optitrack_control_tracking', anonymous = True, disable_signals=True)
    # The dynamic game.
    # Read the parser.
    args= parser.parse_args()
    print(args.ns)
#    main(args)
    robot_game = LQR(args)
#    
#    try:
#        robot_game.control()    
#    except rospy.ROSInterruptException:
#        pass