import time
import rospy
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.armtag import InterbotixArmTagInterface
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface
from dynamic_reconfigure.server import Server
from grab_it.cfg import GrabItConfig
x=0
y=0
z=0
pitch =0


def callback(config, level):
    rospy.loginfo("""Reconfigure Request: {x_param}, {y_param},{z_param}, {pitch_param}""".format(**config))
    bot.arm.set_ee_pose_components(x=x + config.x_param, y=y + config.y_param, z=z+ config.z_param, pitch=pitch + config.pitch_param)
    return config

# This script uses a color/depth camera to get the arm to find objects and pick them up.
# For this demo, the arm is placed to the left of the camera facing outward. When the
# end-effector is located at x=0, y=-0.3, z=0.2 w.r.t. the 'wx200/base_link' frame, the AR
# tag should be clearly visible to the camera. A small basket should also be placed in front of the arm.
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=wx200'
# Then change to this directory and type 'python pick_place.py'



    

 
if __name__=='__main__':
    #rospy.init_node("pick_place", anonymous = False)
    bot = InterbotixManipulatorXS("px150",moving_time=1.5, accel_time=0.75)
    srv = Server(GrabItConfig, callback)
    n = 1
    while n > 0:
	i = 1
