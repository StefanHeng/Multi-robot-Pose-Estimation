#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from util import *

def callback(data):
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data)
    # rospy.loginfo(f'{rospy.get_caller_id()}: {type(data)}')
    # rospy.loginfo(f'{data.header.stamp.secs, type(data.header.stamp)}')
    # rospy.loginfo(f'{data.angle_min, type(data.angle_min)}')
    rospy.loginfo('ran')
    jw(laser_scan2dict(data))

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('Laser Scan Listener', anonymous=True)

    rospy.Subscriber('/hsrb/base_scan', LaserScan, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    fnm = 'HSR laser 2'
    jw = JsonWriter(fnm)

    def _run():
        listener()
    
    def _check():
        with open(f'{fnm}.json') as f:
            l = json.load(f)
            ic(len(l))

    # _run()
    _check()