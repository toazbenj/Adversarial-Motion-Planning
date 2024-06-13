#!/usr/bin/env bash
trap "rosnode kill /Rec" SIGINT
mkdir -p /home/robotics-labs/SGG_DS_ver2/Test$1
#sleep 0.0
gnome-terminal --working-directory=/home/robotics-labs/SGG_DS_ver2/Test$1 -- /bin/bash -c 'rosrun navigation_lab lane_merge_v2.py --r_d "Robot_1" --r_a "Robot_2" --goal_x 2.0 --goal_y 0.0'
gnome-terminal --working-directory=/home/robotics-labs/SGG_DS_ver2/Test$1 -- /bin/bash -c 'rosbag record -a __name:=Rec'
