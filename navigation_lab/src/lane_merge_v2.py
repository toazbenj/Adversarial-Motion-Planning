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

class LM():
    def __init__(self,args):
        # Arguments. 
        self.args = args
        
        # Function to perform when shut down.
        rospy.on_shutdown(self.shutdown)
        #########################
        # Robot parameters.
        #########################
        self.base_vel = 0.18
        #########################
        # Trajectory parameters.
        #########################
        self.goal_x = args.goal_x
        self.goal_x = args.goal_y
        # Horizon.
        self.H = 30
        # Dimension.
        self.n = 2
        # Probability of adversary.
        self.p_ad = 0.1
        ##########################
        # LQR
        ##########################
        # Ros rate.
        self.rosrate = 2.0
        
        # Time step.
        self.dt = 1/self.rosrate
        
        # Discrete time dynamics.
        self.A = [[1.0,0.0],[0.0,1.0]]
        self.B = [[self.dt,0.0],[0.0,self.dt]]
        
        self.A = np.asmatrix(self.A)
        self.B = np.asmatrix(self.B)
        
        # State and control weights.
        self.Q = np.eye((2))
        self.R = 10*np.eye((2))
        
        # Solve the LQR gain.
        self.K = self.dare()
        
        
        #######################
        # PID parameters.
        #######################
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
        self.rate = self.rosrate
        # Reference radius.
        self.r_ref = 1.0
        print(self.args)
        # Subscribe to the adversarial agent.
        rospy.Subscriber("/mocap_node/"+ str(self.args.r_a)+ "/Odom", Odometry, self.adv_agent)
        # Subscribe to defender agent.
        rospy.Subscriber("/mocap_node/"+ str(self.args.r_d)+ "/Odom", Odometry, self.def_agent)
        # Boolean for robots.
        self.ra = False
        self.rf = False
        # Nominal distance between robots.
        self.r_diff = 1.05
        # Create a publisher.
        self.r_vel_d = rospy.Publisher("/" + self.args.r_d + "/cmd_vel", Twist, queue_size = 1)
        self.r_vel_a = rospy.Publisher("/" + self.args.r_a + "/cmd_vel", Twist, queue_size = 1)
        # Counter. 
        self.cnt = 0
        # Game status.
        self.game = True
        # Attacker in the loop.
        self.attack_loop = False
        # Temp.
        self.adv_L_data = np.zeros((2,1))
        self.adv_L_data[0,0] = -1.0
        self.adv_L_data[1,0] = -1.0
        self.def_L_data = np.zeros((2,1))
        self.def_L_data[0,0] = -1.8
        self.def_L_data[1,0] = 0.0
        # Counter.
        self.cnt = 0
        # Attack loop.
        self.attack_loop = False
        # Saving dict.
        self.str_dict = {}
        
    def dare(self):
        '''
        Solve the LQR problem.
        '''
        # Solve ricatti equation.
        X = np.matrix(sp.solve_discrete_are(self.A, self.B, self.Q, self.R))
        # Compute the LQR gain.
        K = -np.matrix(sp.inv(self.B.T*X*self.B + self.R)*(self.B.T*X*self.A))
        
        return K
        
        
    def adv_agent(self, data):
        # Store the data.
        self.adv_data = data
        # Store last data point.
        self.adv_L_data = np.zeros((2,1))
        self.adv_L_data[0,0] = np.copy(data.pose.pose.position.x)
        self.adv_L_data[1,0] = np.copy(data.pose.pose.position.y)
        
    def def_agent(self, data):
        # Store the data.
        self.def_data = data
        
        # Store last data point.
        self.def_L_data = np.zeros((2,1))
        self.def_L_data[0,0] = np.copy(data.pose.pose.position.x)
        self.def_L_data[1,0] = np.copy(data.pose.pose.position.y)
        
        
    def pid_d(self, theta_r, l_nv, yaw):
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

        # Publish the message.
        self.r_vel_d.publish(vel_msg)
        
    def pid_a(self, theta_r, l_nv, yaw):
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

        # Publish the message.
        self.r_vel_a.publish(vel_msg)
        
    def traj_prob(self):
        '''
        Trajectory problem.
        '''
        # Reduced gain.
        self.rg = 0.8
        # Trajectory solution. 
        x_traj = np.zeros((self.n,self.H+1))
        x_traj[1,0] = self.adv_L_data[1,0]
        # Trajectory solution reduced gain.
        x_traj_rg = np.zeros((self.n,self.H+1))
        x_traj_rg[1,0] = self.adv_L_data[1,0]
#        print(self.adv_L_data)
#        print(self.def_L_data)
        # Trajectory of the attacker.
        x_att = np.zeros((self.n,self.H+1))
        x_att[0,0] = self.adv_L_data[0,0]
        x_att[1,0] = self.adv_L_data[1,0]
        # Trajectory of the attacker reduced gain.
        x_att_rg = np.zeros((self.n,self.H+1))
        x_att_rg[0,0] = self.adv_L_data[0,0]
        x_att_rg[1,0] = self.adv_L_data[1,0]
        # Trajectory solution defender. 
        x_traj_def = np.zeros((self.n,self.H+1))
        x_traj_def[1,0] = self.def_L_data[1,0]
        # Trajectory solution defender gain.
        x_traj_def_rg = np.zeros((self.n,self.H+1))
        x_traj_def_rg[1,0] = self.def_L_data[1,0]
        
        # Trajectory of the defender.
        x_def = np.zeros((self.n,self.H+1))
        x_def[0,0] = self.def_L_data[0,0]
        x_def[1,0] = self.def_L_data[1,0]
        # Trajectory of the defender reduced gain.
        x_def_rg = np.zeros((self.n,self.H+1))
        x_def_rg[0,0] = self.def_L_data[0,0]
        x_def_rg[1,0] = self.def_L_data[1,0]
        # Store control policy.
        self.att_vel = np.zeros((2,self.H+1))
        self.att_vel_rg = np.zeros((2,self.H+1))
        self.def_vel = np.zeros((2,self.H+1))
        self.def_vel_rg = np.zeros((2,self.H+1))
        
#        print(self.adv_L_data)
#        print(self.def_L_data)
        for t in range(self.H):
#            temp  = np.matmul(self.A + np.matmul(self.B,self.K*0.8),x_traj[:,t])
#            print(np.array(temp).flatten().shape)
            # LQR trajectory.
            x_traj[:,t+1] = np.array(np.matmul(self.A + np.matmul(self.B,self.K),x_traj[:,t])).flatten()
            # Attack trajectory.
            x_att[0,t+1] = x_att[0,t] + self.dt*self.base_vel
            x_att[1,t+1] = x_traj[1,t+1]
            # Velocity.\
            u_att = np.matmul(self.B,np.matmul(self.K,x_traj[:,t]).T)
            self.att_vel[1,t] = u_att[1,0]
            self.att_vel[0,t] = self.base_vel
            # LQR trajectory reduced gain.
            x_traj_rg[:,t+1] = np.array(np.matmul(self.A + np.matmul(self.B,self.K*self.rg),x_traj[:,t])).flatten()
            # Attack trajectory.
            x_att_rg[0,t+1] = x_att[0,t] + self.dt*self.base_vel*self.rg
            x_att_rg[1,t+1] = x_traj_rg[1,t+1]
            # Velocity.
            u_att = np.matmul(self.B,np.matmul(self.K*self.rg,x_traj[:,t]).T)
            self.att_vel_rg[1,t] = u_att[1,0]
            self.att_vel_rg[0,t] = self.base_vel*self.rg
            
            
            #######################
            # Defender trajectory.
            #######################
            # LQR trajectory.
            x_traj_def[:,t+1] = np.array(np.matmul(self.A + np.matmul(self.B,self.K),x_traj_def[:,t])).flatten()
            x_def[1,t+1] = x_traj_def[1,t+1]
            x_def[0,t+1] = x_def[0,t] + self.dt*self.base_vel
            # Velocity.
            u_def = np.matmul(self.B,np.matmul(self.K,x_traj_def[:,t]).T)
            self.def_vel[1,t] = u_def[1,0]
            self.def_vel[0,t] = self.base_vel
            
            # Defender trajectory reduced gain.
            # LQR trajectory reduced gain.
            x_traj_def_rg[:,t+1] = np.array(np.matmul(self.A + np.matmul(self.B,self.K*self.rg),x_traj_def[:,t])).flatten()
            x_def_rg[1,t+1] = x_traj_def_rg[1,t+1]
            x_def_rg[0,t+1] = x_def[0,t] + self.dt*self.base_vel*self.rg
            # Velocity.
            u_def = np.matmul(self.B,np.matmul(self.K*self.rg,x_traj_def_rg[:,t]).T)
            self.def_vel_rg[1,t] = u_def[1,0]
            self.def_vel_rg[0,t] = self.base_vel*0.8
        # Trajectories.
        self.x_att = x_att
        self.x_att_rg = x_att_rg
        self.x_def = x_def
        self.x_def_rg = x_def_rg
        
        # Final location.
        self.x_final = x_def[:,-1]
        # Cost to go ffrom stopping state.
        ctg_ss = np.zeros((1,self.H))
        ctg_stg = np.zeros((1,self.H))
        
        for t in range(self.H-1,-1,-1):
            temp = x_def_rg[:,t+1] - self.x_final
            ctg_stg[0,t] = np.matmul(np.matmul(temp.T,self.Q),temp)
            
        for t in range(self.H):
            ctg_ss[0,t] = np.sum(ctg_stg[0,t:])
#        pdb.set_trace()
        # Unsafe distance.
        self.un_d = 1.0*np.linalg.norm(x_att[:,0] - x_def[:,0])
        
        ##############################
        # Stochastic game.
        #############################
        # Policy.
        def player(X):
            # policy.
            y = np.zeros((2,1))
            y[0,0] = (X[1,1] - X[1,0])/(X[0,0] - X[0,1] - X[1,0] + X[1,1])
            y[1,0] = 1 - y[0,0]
            
            return y
        # Value of the game. 
        BV_s = np.zeros((self.H+1,1))
        
        # Initial stage cost.
        V = 0
        # Cost to go for yield state. 
        CTG = 0
        
        for stg in range(self.H-1,-1,-1):
            # Log factor.
            fc = 5.0
            # Quadratic factor.
            qc = 1.0
            # Current stage costs.
            S = np.zeros((self.n,self.n))
#            pdb.set_trace()
            # Def - Yield, Att - Move,
            LQR_traj = x_def_rg[:,stg+1] - self.x_final
            temp = np.linalg.norm(x_att[:,stg+1] - x_def_rg[:,stg+1])
            S[0,0] = np.matmul(np.matmul(LQR_traj.T,self.Q),LQR_traj) - fc*np.log(temp/self.un_d) + ctg_ss[0,stg]
#            pdb.set_trace()
            # Def - Yield, Att - Yield.
            LQR_traj = x_def_rg[:,stg+1] - self.x_final
            temp = np.linalg.norm(x_att_rg[:,stg+1] - x_def_rg[:,stg+1])
            S[0,1] = np.matmul(np.matmul(LQR_traj.T,self.Q),LQR_traj) - fc*np.log(temp/self.un_d)
            
            # Def - Move, Att - Move.
            LQR_traj = x_def[:,stg+1] - self.x_final
            temp = np.linalg.norm(x_att[:,stg+1] - x_def[:,stg+1])
            S[1,0] = np.matmul(np.matmul(LQR_traj.T,self.Q),LQR_traj) - fc*np.log(temp/self.un_d)
            
            # Def - Move, Att - Yield.
            LQR_traj = x_def[:,stg+1] - self.x_final
            temp = np.linalg.norm(x_att_rg[:,stg+1] - x_def[:,stg+1])
            S[1,1] = np.matmul(np.matmul(LQR_traj.T,self.Q),LQR_traj) - fc*np.log(temp/self.un_d)
#            print(S)
            # First matrix.
            D = V*np.ones((2,2))
            D[0,0] = 0.0
            CTG_M = np.zeros((2,2))
            CTG_M[0,1] = CTG
            A = D + CTG_M + S
            
            # Difference between yield and move cost to go.
            diff_CTG = x_def[:,stg+1] - x_def_rg[:,stg+1]
            CTG = np.matmul(np.matmul(diff_CTG.T,self.Q),diff_CTG)
            
            Vover = np.min(np.max(A,axis=1))
            y = np.argmin(np.max(A,axis=1))
            Vunder = np.max(np.min(A,axis=0))
            z = np.argmax(np.min(A,axis=0))
            
            if Vover!=Vunder:
                # Policy for defender. 
                p_def = player(np.copy(A))
                # Policy for attacker. 
                p_att = player(A.T)
                # Value of the game.
                V = np.matmul(np.matmul(p_def.T,A),p_att) 
                BV_s[stg,0] = V
            else:
                BV_s[stg,0] = Vover
                V = Vover
            
        # Modified structure.
        # Probability of attacker.
        self.p_itr = 0.1
        # Value of the game
        BV_M_s = np.zeros((self.H,1))
        
        # Policy of the move state.
        def_pol = np.zeros((self.H,2))
        att_pol = np.zeros((self.H,2))
        
        # Value of the initial stage. 
        V_M = 0
        # Difference value state
        CTG = 0
        
        for stg in range(self.H-1,-1,-1):
#            print(stg)
            # Current stage costs.
            S = np.zeros((self.n,self.n))
            # Def - Yield, Att - Move,
            LQR_traj = x_def_rg[:,stg+1] - self.x_final
            temp = np.linalg.norm(x_att[:,stg+1] - x_def_rg[:,stg+1])
            S[0,0] = np.matmul(np.matmul(LQR_traj.T,self.Q),LQR_traj) - fc*np.log(temp/self.un_d) + ctg_ss[0,stg]
            
            # Def - Yield, Att - Yield.
            LQR_traj = x_def_rg[:,stg+1] - self.x_final
            temp = np.linalg.norm(x_att_rg[:,stg+1] - x_def_rg[:,stg+1])
            S[0,1] = np.matmul(np.matmul(LQR_traj.T,self.Q),LQR_traj) - fc*np.log(temp/self.un_d)
            
            # Def - Move, Att - Move.
            LQR_traj = x_def[:,stg+1] - self.x_final
            temp = np.linalg.norm(x_att[:,stg+1] - x_def[:,stg+1])
            S[1,0] = np.matmul(np.matmul(LQR_traj.T,self.Q),LQR_traj) - fc*np.log(temp/self.un_d)
            
            # Def - Move, Att - Yield.
            LQR_traj = x_def[:,stg+1] - self.x_final
            temp = np.linalg.norm(x_att_rg[:,stg+1] - x_def[:,stg+1])
            S[1,1] = np.matmul(np.matmul(LQR_traj.T,self.Q),LQR_traj) - fc*np.log(temp/self.un_d)
            
            # First matrix.
            D = BV_s[stg+1,0]*np.ones((2,2))
            D[0,0] = 0.0
            CTG_M = np.zeros((2,2))
            CTG_M[0,1] = CTG
            A = D + CTG_M + S
            
            # Second matrox.
            B = np.zeros((2,2))
            B[0,0] = S[0,1]
            B[0,1] = S[0,1]
            B[1,0] = S[1,1]
            B[1,1] = S[1,1]
            
            D2 = V_M*np.ones((2,2))
            D2[0,0] = D2[0,0] + CTG
            D2[0,1] = D2[0,1] + CTG
            
            B = B + D2
            
            
            # Difference between yield and move cost to go.
            diff_CTG = x_def[:,stg+1] - x_def_rg[:,stg+1]
            CTG = np.matmul(np.matmul(diff_CTG.T,self.Q),diff_CTG)
            
            # Determine threshold.
            p_th = (A[1,1]-A[0,1])/(A[0,0] - A[0,1] - A[1,0] + A[1,1])
            
            if self.p_itr >= p_th:
                
            
                Vover = np.min(np.max(self.p_itr*A + (1-self.p_itr)*B,axis=1))
                y = np.argmin(np.max(self.p_itr*A + (1-self.p_itr)*B,axis=1))
                Vunder = np.max(np.min(self.p_itr*A + (1-self.p_itr)*B,axis=0))
                z = np.argmax(np.min(self.p_itr*A + (1-self.p_itr)*B,axis=0))
            
                if Vover!=Vunder:
                    # Policy for defender. 
                    p_def = player(np.copy(A))
                    def_pol[stg,:] = p_def.flatten()
                    # Policy for attacker. 
                    p_att = player(A.T)
#                    pdb.set_trace()
                    if p_att[1,0]!=1:
                        p_att[0,0] = p_att[0,0]/self.p_itr
                        p_att[1,0] = 1 - p_att[0,0]
                    att_pol[stg,:] = p_att.flatten()
                    # Value of the game.
                    V_M = np.matmul(np.matmul(p_def.T,self.p_itr*A + (1-self.p_itr)*B),p_att)
                    BV_M_s[stg,0] = V_M
                else:
                    if y==0:
                        p_def = np.array(([1],[0]))
                        
                    else:
                        p_def = np.array(([0],[1]))
                        
                    if z==0:
                        p_att = np.array(([1],[0]))
                    else:
                        p_att = np.array(([0],[1]))
                        
                    def_pol[stg,:] = p_def.flatten()
                    att_pol[stg,:] = p_att.flatten()
                    BV_M_s[stg,0] = Vover
                    V_M = Vover
            else:
                # Defender policy.
                p_def = np.array(([0],[1]))
                
                # Attacker policy.
                p_att = np.array(([1],[0]))
                
                def_pol[stg,:] = p_def.flatten()
                att_pol[stg,:] = p_att.flatten()
                
                # Value of the game.
                V_M = np.matmul(np.matmul(p_def.T,self.p_itr*A + (1-self.p_itr)*B),p_att)
                
                BV_M_s[stg,0] = V_M
#            if stg==0:
#                print('hold')
#                pdb.set_trace()
                
        # Store.
        self.BV_M_s = BV_M_s
        self.def_pol = def_pol
        self.att_pol = att_pol
            
        
            
            
        
        
    def control(self):
        '''
        Loop where we will do the current control.
        
        '''
        # Set a dummy callback.
        # Adversarial agent.
        self.adv_data = Odometry()
        # Defender agent.
        self.def_data = Odometry()
        # Set the rate of control loop. 
        rate = rospy.Rate(self.rosrate)
        # Solve the trajectory problem.
        self.traj_prob()
#        print(self.BV_M_s)
        
            
        while not rospy.is_shutdown():
            # Update dictionary.
            self.str_dict[self.cnt] = {}
            
            # Get the current position of adversary and defender agent.
            rospy.Subscriber("/mocap_node/"+ str(self.args.r_a)+ "/Odom", Odometry, self.adv_agent)
            rospy.Subscriber("/mocap_node/"+ str(self.args.r_d)+ "/Odom", Odometry, self.def_agent)
            
            
            # Solve the trajectory problem.
            self.traj_prob()
            # Trajectory.
            self.str_dict[self.cnt]['x_att'] = self.x_att
            self.str_dict[self.cnt]['x_att_rg'] = self.x_att_rg
            self.str_dict[self.cnt]['x_def'] = self.x_def
            self.str_dict[self.cnt]['x_def_rg'] = self.x_def_rg
            
            self.str_dict[self.cnt]['att_vel'] = self.att_vel
            self.str_dict[self.cnt]['att_vel_rg'] = self.att_vel_rg
            self.str_dict[self.cnt]['def_vel'] = self.def_vel
            self.str_dict[self.cnt]['def_vel_rg'] = self.def_vel_rg
            
            self.str_dict[self.cnt]['def_pol'] = self.def_pol
            self.str_dict[self.cnt]['att_pol'] = self.att_pol
            self.str_dict[self.cnt]['BV_M_s'] = self.BV_M_s
            
            if self.game == True:
#                print(self.att_pol)
                # Sample the policy. 
                att_action = np.random.binomial(1,self.att_pol[0,0])
                def_action = np.random.binomial(1,self.def_pol[0,0])
                
                # Save the policy.
                self.str_dict[self.cnt]['c_att_pol'] = att_action
                self.str_dict[self.cnt]['c_def_pol'] = def_action
                
                # Sample the presence of an attacker.
                if self.attack_loop == False:
                    prs_attack = np.random.binomial(1,self.p_ad)
                    if prs_attack == 0:
                        att_action = 0
                    else:
                        self.attack_loop = True
#                print(self.att_pol[0,0])
#                print(self.def_pol[0,0])
                print('Attack ' + str(att_action))
                print('Defend ' + str(def_action))
                if att_action == 1 and def_action==1:
                    self.game = False
                    
                if att_action == 1:
                    # Set velocity for the robot.
                    a_vel_ref = np.linalg.norm(self.att_vel[:,0])
                    a_ang_ref = np.arctan2(self.att_vel[1,0],self.att_vel[0,0])
                else:
                    # Set velocity for the robot.
                    a_vel_ref = np.linalg.norm(self.att_vel_rg[:,0])
                    a_ang_ref = np.arctan2(self.att_vel_rg[1,0],self.att_vel_rg[0,0])
                
                if def_action == 1:
                    # Set velocity for the robot.
                    d_vel_ref = np.linalg.norm(self.def_vel_rg[:,0])
#                    d_vel_ref = self.def_vel_rg[0,0]
                    d_ang_ref = np.arctan2(self.def_vel_rg[1,0],self.def_vel_rg[0,0])
#                    d_ang_ref = 0.0
                else:
                    # Set velocity for the robot.
                    d_vel_ref = np.linalg.norm(self.def_vel[:,0])
#                    d_vel_ref = self.def_vel[0,0]
                    d_ang_ref = np.arctan2(self.def_vel[1,0],self.def_vel[0,0])
#                    d_ang_ref = 0.0
            else:
                # Game over.
                print('Game over')
                # Defender velocity.
                d_vel_ref = np.linalg.norm(self.def_vel_rg[:,0])
#                d_vel_ref = self.def_vel_rg[0,0]
                d_ang_ref = np.arctan2(self.def_vel_rg[1,0],self.def_vel_rg[0,0])
#                d_ang_ref = 0.0
                # Attacker velocity.
                a_vel_ref = np.linalg.norm(self.att_vel[:,0])
                a_ang_ref = np.arctan2(self.att_vel[1,0],self.att_vel[0,0])
                
                
            # Angular data adversary.
            orientation_q = self.adv_data.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw_a) = euler_from_quaternion(orientation_list)
            
            # Call PID for angular control.
            self.pid_a(a_ang_ref, a_vel_ref, yaw_a)
            
            # Angular data adversary.
            orientation_q = self.def_data.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw_d) = euler_from_quaternion(orientation_list)
            
            # Call PID for angular control.
            self.pid_d(d_ang_ref, d_vel_ref, yaw_d)
            print(d_ang_ref)
            
            self.cnt = self.cnt + 1
            
            if self.cnt >= 40:
                rospy.signal_shutdown("Reactor shutting down.")
            
            rate.sleep()
                
        
        
    def shutdown(self):
        print('Begin shutdown')
        # Save the data.
#        dir_s = '/home/robotics-labs/SGG_SS/Test'
#        with open(dir_s + self.args.sd + 'position_ver2.npy', 'wb') as f:
#            np.save(f, np.array(self.pos_data))
        
        with open('data_dictionary.pkl', 'wb') as f:
            pickle.dump(self.str_dict, f)
        r_vel_d = rospy.Publisher("/" + self.args.r_d + "/cmd_vel", Twist, queue_size = 1)
        r_vel_a = rospy.Publisher("/" + self.args.r_a + "/cmd_vel", Twist, queue_size = 1)
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0
        r_vel_d.publish(vel_msg)
        r_vel_a.publish(vel_msg)
        # Command to stop rosbag recording.
        os.system('rosnode kill /Rec')
        
parser = argparse.ArgumentParser(description = "Rotary Control follower")
parser.add_argument("--r_d", type=str, default=None, help="Robot defender")
parser.add_argument("--r_a", type=str, default=None, help="Robot attacker")
parser.add_argument("--goal_x", type=str, default=None, help="Goal location x")
parser.add_argument("--goal_y", type=str, default=None, help="Goal location y")

        

if __name__ == '__main__':
    print('Start')
    # Initialize the node.
    rospy.init_node('lane_merge', anonymous = True, disable_signals=True)
    # Read the parser.
    args= parser.parse_args()
    print(args)
    robot_game = LM(args)
    
    
    try:
        robot_game.control()    
    except rospy.ROSInterruptException:
        pass