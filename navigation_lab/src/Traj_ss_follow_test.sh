#!/usr/bin/env bash
trap "rosnode kill /Rec" SIGINT
mkdir -p /home/robotics-labs/SGG_SS/Test$1
sleep 7.0
#gnome-terminal -- /bin/bash -c 'rosrun navigation_lab optitrack_control_tracking.py --ns "Robot_1" --x_loc 0.0 --y_loc 1.0 --tht 3.14' 
gnome-terminal -- /bin/bash -c 'rosrun navigation_lab straight_traj.py --ns "Robot_2"'
#gnome-terminal -- /bin/bash -c 'rosrun navigation_lab optitrack_control_tracking.py --ns "Robot_2" --x_loc 0.5 --y_loc 0.866 --tht 2.094'
gnome-terminal --working-directory=/home/robotics-labs/SGG_SS/Test$1 -- /bin/bash -c 'rosrun navigation_lab straight_traj_follower_v2.py --r_f "Robot_1" --r_a "Robot_2"'
gnome-terminal --working-directory=/home/robotics-labs/SGG_SS/Test$1 -- /bin/bash -c 'rosbag record -a __name:=Rec'
#gnome-terminal --working-directory=/home/robotics-labs/SGG_SS/Test$1 -- /bin/bash -c 'rosrun navigation_lab rosbag_record.py'
