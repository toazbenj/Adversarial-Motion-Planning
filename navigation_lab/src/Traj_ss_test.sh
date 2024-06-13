#!/usr/bin/env bash
#gnome-terminal -- /bin/bash -c 'rosrun topic_tools throttle messages /mocap_node/Robot_1/Odom 20.0' 
#gnome-terminal -- /bin/bash -c 'rosrun navigation_lab optitrack_control_tracking.py --ns "Robot_1" --x_loc -1.8 --y_loc 0.0 --tht 0.0' 
# Dynamic trajectory.
gnome-terminal -- /bin/bash -c 'rosrun navigation_lab optitrack_control_tracking.py --ns "Robot_1" --x_loc -1.8 --y_loc 0.0 --tht 0.0' 


#gnome-terminal -- /bin/bash -c 'rosrun navigation_lab rotary_traj.py --ns "Robot_1"'
#gnome-terminal -- /bin/bash -c 'rosrun topic_tools throttle messages /mocap_node/Robot_2/Odom 20.0'  
#gnome-terminal -- /bin/bash -c 'rosrun navigation_lab optitrack_control_tracking.py --ns "Robot_2" --x_loc -0.8 --y_loc 0.0 --tht 0.0'
# Dynamic trajectory.
gnome-terminal -- /bin/bash -c 'rosrun navigation_lab optitrack_control_tracking.py --ns "Robot_2" --x_loc -1.0 --y_loc -1.0 --tht 0.0'
#gnome-terminal -- /bin/bash -c 'rosrun navigation_lab rotary_traj_follower.py --r_f "Robot_2" --r_a "Robot_1"'