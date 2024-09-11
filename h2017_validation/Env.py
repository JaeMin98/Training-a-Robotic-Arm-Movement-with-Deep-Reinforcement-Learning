#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import rospy
import moveit_commander #Python Moveit interface를 사용하기 위한 모듈
import moveit_msgs.msg
import geometry_msgs.msg
import math
from moveit_commander.conversions import pose_to_list
from moveit_msgs.msg import RobotState
from tf.transformations import quaternion_matrix
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import math
import time
import numpy as np
from sensor_msgs.msg import JointState
import csv
import random

class Ned2_control(object):
    def __init__(self):
        super(Ned2_control, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        group_name = "h2017"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        self.robot = moveit_commander.RobotCommander()
        self.move_group = move_group

        self.target = [0,0,0]

        # CSV 파일 열기
        self.level_point = []
        for i in range(1,6):
            with open('./DataCSV/UoC_'+str(i)+'.csv', 'r') as file:
                reader = csv.reader(file)

                # 각 행들을 저장할 리스트 생성
                rows = []

                for row in reader:
                    row_temp = row[:3]
                    rows.append(row_temp)
                self.level_point.append(rows)

        self.MAX_Level_Of_Point = 4
        self.Level_Of_Point = 0

        self.isLimited = False
        self.Iswait = True

        self.Limit_joint=[[-180.0,180.0],
                            [-110.0,110.0],
                            [-140.0,140.0],
                            [-0.1,0.1],
                            [-0.1,0.1],
                            [-0.1,0.1]]
        
        self.weight = [20, 20, 20,0, 0, 0]

        self.time_step = 0
        self.MAX_time_step = 200

        self.prev_distance = None
        self.collision_detected = False

    def Degree_to_Radian(self,Dinput):
        return np.array(Dinput) * (np.pi / 180.0)

    def Radian_to_Degree(self,Rinput):
        return np.array(Rinput) * (180.0 / np.pi)
    
    def calc_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def action(self,angle):  # angle 각도로 이동 (angle 은 크기 6의 리스트 형태)
        joint = self.move_group.get_current_joint_values()
        angle = self.Degree_to_Radian(angle)

        joint[0] += (angle[0]) * self.weight[0]
        joint[1] += (angle[1]) * self.weight[1]
        joint[2] += (angle[2]) * self.weight[2]
        joint[3] = 0
        joint[4] = 0
        joint[5] = 0

        for i in range(len(self.Limit_joint)):
            if(self.Limit_joint[i][1] < joint[i]):
                joint[i] = self.Limit_joint[i][1]
            elif(self.Limit_joint[i][0] > joint[i]):
                joint[i] = self.Limit_joint[i][0]

        try:
            plan = self.move_group.go(joint, wait=self.Iswait)
        except:
            plan = False

        self.collision_detected = not plan
        self.time_step += 1
            
    def reset(self):
        self.time_step = 0
        self.prev_distance = None
        self.isLimited = False
        self.move_group.go([0,0,0,0,0,0], wait=True)


        random_index_list = random.choice(range(len(self.level_point[self.Level_Of_Point])))
        self.target = self.level_point[self.Level_Of_Point][random_index_list]
        self.target = [float(element) for element in self.target] # 목표 지점 위치
        self.target_reset()

        
    
    def get_endeffector_position(self):
        pose = self.move_group.get_current_pose().pose
        pose_value = [pose.position.x,pose.position.y,pose.position.z]
        return pose_value
    
    def get_state(self):
        joint = self.move_group.get_current_joint_values()
        state = joint[0:3] + self.target
        return state

    def get_reward(self):
        distance = self.calc_distance(self.get_endeffector_position(), self.target)

        R_basic= -1 * distance
        R_done= 0
        R_extra = 0
        if (self.prev_distance != None ): R_extra = -1 * (distance - self.prev_distance)
        self.prev_distance = distance

        isDone, isSuccess = False, False

        if(self.collision_detected):
            R_done = -50
            isDone,isSuccess = True, False
     
        if(self.time_step >= self.MAX_time_step):
            R_done = 0
            isDone,isSuccess = True, False

        if(distance <= 0.2):
            R_done = 50
            isDone,isSuccess = True,True

        totalReward = R_basic + R_done + R_extra
        return totalReward, isDone,isSuccess
    
    def step(self, angle):
        self.action(angle)
        self.move_group.stop()

        totalReward,isDone,isSuccess = self.get_reward()
        current_state = self.get_state()

        return current_state,totalReward,isDone, isSuccess
    

    def target_reset(self):
        state_msg = ModelState()
        state_msg.model_name = 'cube'
        state_msg.pose.position.x = self.target[0]
        state_msg.pose.position.y = self.target[1]
        state_msg.pose.position.z = self.target[2]
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        for i in range(100):
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            resp = set_state(state_msg)    


if __name__ == "__main__":
    ned2_control = Ned2_control()
    rospy.sleep(1)  # 초기화 시간 대기

    # # 테스트 코드
    ned2_control.reset()

    while not rospy.is_shutdown():
        ned2_control.step([0.0, -0.3, 0.35, 0, 0, 0])
        print(ned2_control.get_joint3_position())
