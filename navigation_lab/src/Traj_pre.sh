#!/usr/bin/env bash
gnome-terminal -- /bin/bash -c 'rosrun navigation_lab optitrack_control_tracking.py --ns "Robot_2" --x_loc -1.0 --y_loc -1.0' 
#gnome-terminal -- /bin/bash -c 'rosrun navigation_lab rotary_traj.py --ns "Robot_1"'
gnome-terminal -- /bin/bash -c 'rosrun navigation_lab optitrack_control_tracking.py --ns "Robot_1" --x_loc -1.0 --y_loc 1.0'
#gnome-terminal -- /bin/bash -c 'rosrun navigation_lab rotary_traj_follower.py --r_f "Robot_2" --r_a "Robot_1"'