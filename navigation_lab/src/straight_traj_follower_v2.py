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
        # PID
        ################
        self.args = args
        # Proportional gain.
        self.Kp = 0.5*1.0
        # Derivative gain.
        self.Kd = 0.5*2.0
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
        # Solve the policy.
        # self.SSG()
        ######################
        # Subscribe to the agent ahead.
        self.agent_a = rospy.Subscriber("/mocap_node/"+ str(self.args.r_a)+ "/Odom", Odometry, self.robot_ahead)
        # Subscribe to current robot.
        self.agent_self = rospy.Subscriber("/mocap_node/"+ str(self.args.r_f)+ "/Odom", Odometry, self.robot_following)
        # Boolean for robots.
        self.ra = False
        self.rf = False
        # Nominal distance between robots.
        self.r_diff = 1.05
        # Create a publisher.
        #######################
        self.r_vel = rospy.Publisher("/" + self.args.r_f + "/cmd_vel", Twist, queue_size = 1)
        self.ra_vel = rospy.Publisher("/" + self.args.r_a + "/cmd_vel", Twist, queue_size = 1)
        #######################
        # Counter. 
        self.cnt = 0
        # Game status.
        self.game = True
        # Attacker in the loop.
        self.attack_loop = False
        # Store sample policy.
        self.str_pol_att = []
        self.str_pol_def = []
        # Dictionary for saving.
        self.str_dict = {}
        
    def SSG(self):
        '''
        Stochastic stopping state game.
        '''
        # Number of stages. 
        K = int(np.ceil(3.5/0.14))
        self.K = K
        # Defend factor
        phi_1 = (3.5/0.12/K)
        # Attack factor.
        phi_a = (3.5/0.10/K)
        
        # Stage costs. 
        S = np.zeros((2,2))
        S[0,0] = phi_1 + (0.14-0.12)
        S[0,1] = phi_1 + (0.14-0.12)
        S[1,0] = phi_a + (0.10-0.12)
        S[1,1] = 1.0
        print(S)
        # Policy.
        def player(X):
            # policy.
            y = np.zeros((2,1))
            y[0,0] = (X[1,1] - X[1,0])/(X[0,0] - X[0,1] - X[1,0] + X[1,1])
            y[1,0] = 1 - y[0,0]
            
            return y
        
        # Initial stage cost.
        V = 0
        VG_str = np.zeros((K+1,1))
        # Save policy.
        def_pol = np.zeros((K,2))
        att_pol = np.zeros((K,2))
        # Decision matrix.
        DM = np.ones((2,2))
        DM[0,0] = 0.0
        for i in range(K,0,-1):
            # Matrix Val.
            A = V*DM + S
            # Policy for defender. 
            p_def = player(np.copy(A))
            def_pol[i-1,:] = p_def.flatten()
            # Policy for attacker. 
            p_att = player(A.T)
            att_pol[i-1,:] = p_att.flatten()
            # Value of the game.
            V = np.matmul(np.matmul(p_def.T,A),p_att) 
            VG_str[i-1,0] = V
            
        # Stochastic games.
        # Probability of attack.
        p_a = 0.25
        self.p_a = p_a
        # Initial stage cost.
        V = 0# Create a publisher.
        self.r_vel = rospy.Publisher("/" + self.args.r_f + "/cmd_vel", Twist, queue_size = 1)
        
        SVG_str = np.zeros((K,1))
        # Save policy.
        SG_def_pol = np.zeros((K,2))
        SG_att_pol = np.zeros((K,2))
        # Decision matrix.
        DM = np.ones((2,2))
        DM[0,0] = 0.0
        for i in range(K,0,-1):
            # Matrix Val.
            A = VG_str[i]*DM + S
            # Matrix B.
            B = np.array(([S[0,1], S[0,1]],[S[1,1], S[1,1]])) + V
#            pdb.set_trace()
            # Determine threshold.
            p_th = (A[1,1]-A[0,1])/(A[0,0] - A[0,1] - A[1,0] + A[1,1])
            if p_a >= p_th:
                # Policy for defender. 
                p_def = player(np.copy(A))
                SG_def_pol[i-1,:] = p_def.flatten()
                # Policy for attacker. 
                p_att = player(A.T)
                p_att[0,0] = p_att[0,0]/p_a
                p_att[1,0] = 1 - p_att[0,0]
                SG_att_pol[i-1,:] = p_att.flatten()
                # Value of the game.
                V = p_a*np.matmul(np.matmul(p_def.T,A),p_att) + (1-p_a)*np.matmul(np.matmul(p_def.T,B),p_att)
                SVG_str[i-1,0] = V
            else:
                p_def = np.array(([0.0,1.0]))
                SG_def_pol[i-1,:] = p_def
                # Policy for attacker. 
                p_att = np.array(([1.0,0.0]))
                SG_att_pol[i-1,:] = p_att
                # Value of the game.
                V = np.matmul(np.matmul(p_def.T,A),p_att) 
                SVG_str[i-1,0] = V
                
        # Store the policy.
        self.SG_def_pol = SG_def_pol
        self.SG_att_pol = SG_att_pol
        self.SVG_str = SVG_str
        print(SVG_str)
                
    def robot_ahead(self,data):
        self.data_a = data
    
    def robot_following(self,data):
        self.data_f = data
        
    def test(self,data):
        print(data)
        
    def control(self):
        
        # Set the rate.
        r = rospy.Rate(1.0)
        
        # Create empty data.
        self.data_a = Odometry()
        self.data_f = Odometry()
        
        
        # Get data from topic.
        rospy.Subscriber("/mocap_node/"+ str(self.args.r_a)+ "/Odom", Odometry, self.robot_ahead)
        rospy.Subscriber("/mocap_node/"+ str(self.args.r_f)+ "/Odom", Odometry, self.robot_following)
        
        
        while not rospy.is_shutdown():
            
            '''
            print(self.SG_att_pol[self.cnt,0])
            if self.data_a.header.stamp.secs != 0:
                # Position of the robot ahead.
                pos_ahead = self.data_a.pose.pose.position
                self.ra = True
            else:
                self.ra = False
                
            if self.data_f.header.stamp.secs != 0:
                # Position of the robot ahead.
                pos_self = self.data_f.pose.pose.position
                self.rf = True
            else:
                self.rf = False
            
            
            
            if self.rf and self.ra:
                # Determine the difference between robots.
                self.r_diff = np.linalg.norm(np.array([pos_self.x - pos_ahead.x,pos_self.y - pos_ahead.y]))
                
                d_theta = np.arctan2((pos_ahead.y - pos_self.y),(pos_ahead.x - pos_self.x))
            else:
                d_theta = 0.0
                
            
            # Game.
            # Sample policy. 
            att_action = np.random.binomial(1,self.SG_att_pol[self.cnt,0])
            def_action = np.random.binomial(1,self.SG_def_pol[self.cnt,0])
            print('Attack ' + str(att_action))
            self.str_pol_att.append(att_action)
            print('Defend ' + str(def_action))
            self.str_pol_def.append(def_action)
            # Sample the presence of an attacker.
            if self.attack_loop == False:
                prs_attack = np.random.binomial(1,self.p_a)
                if prs_attack == 0:
                    att_action = 0
                else:
                    self.attack_loop = True
                
            if att_action == 1 and def_action == 1:
                print('Game over')
                self.game = False
            
            # Default velocity.
            vel_l = 0.14
            
            if self.game == True:
                if att_action == 1:
                    self.r_diff = 0.10/0.14
                    vel_l = 0.1
                elif def_action ==1:
                    self.r_diff = 0.12/0.14
                    vel_l = 0.12
            else:
                vel_l = 0.14
                
#            # Determine policy. 
#            if self.r_diff <= 1.05:
#                vel_l = 0.12
#            else:
#                vel_l = 0.14
            # PID.
            print(self.r_diff)
            # Calculate the error.
            err_theta = d_theta
#            print(d_theta)
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
            
            '''
            # Blank message.
            # Use self.data_a and self.data_f to make a rule.
            # Read configuration files of Turtlebot for max speed and min speed. 

            if dist <= 0.5:
                vel_r = 0.15
            else:
                vel_r = 0.18

            vel_msg = Twist()
            vel_msg.linear.y = 0.0
            vel_msg.linear.z = 0.0
            
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            
            vel_msg.angular.z = 0.0
            vel_msg.linear.x = Ishwari_input
#            print(Kp + Kd)
            # Publish the message.
            self.r_vel.publish(vel_msg)
            
            self.cnt = self.cnt + 1
            
            if self.cnt >= self.K:
                rospy.signal_shutdown("Reactor shutting down.")
            
            r.sleep()
        
        
       
        
    def shutdown(self):
        print('Begin shutdown')
        '''
        # Save the data.
        dir_s = os.getcwd()
        with open(dir_s + '/att_pol.npy', 'wb') as f:
            np.save(f, np.array(self.str_pol_att))
        with open(dir_s + '/def_pol.npy', 'wb') as f:
            np.save(f, np.array(self.str_pol_def))
        ''' 
        r_vel = rospy.Publisher("/" + self.args.r_f + "/cmd_vel", Twist, queue_size = 1)
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0
        r_vel.publish(vel_msg)
        #os.system('rosnode kill /Rec')

parser = argparse.ArgumentParser(description = "Rotary Control follower")
parser.add_argument("--r_f", type=str, default="Tb3_1", help="Robot following")
parser.add_argument("--r_a", type=str, default="Tb3_0", help="Robot ahead")
        
#def main(args):
#
#    robot_game = Follower(args)
#    r = rospy.Rate(1)
#    while not rospy.is_shutdown():
#        r.sleep()

if __name__ == '__main__':
    print('Start')
    # Initialize the node.
    rospy.init_node('straight_traj_follower_v2', anonymous = True, disable_signals=True)
    # Read the parser.
    args= parser.parse_args()
    robot_game = Follower(args)
    
    
    try:
        robot_game.control()    
    except rospy.ROSInterruptException:
        pass
    