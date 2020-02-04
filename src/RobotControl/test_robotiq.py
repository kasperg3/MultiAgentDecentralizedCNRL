from RobotControl.robotiq_gripper import RobotiqGripper
from rtde_control import RTDEControlInterface
import time

rtde_c = RTDEControlInterface("24.5.19.10")
gripper = RobotiqGripper(rtde_c)

# Activate the gripper and initialize force and speed
gripper.activate()  # returns to previous position after activation
gripper.set_force(50)  # from 0 to 100 %
gripper.set_speed(100)  # from 0 to 100 %

# Perform some gripper actions
gripper.open()
gripper.move(30)  # mm

# Stop the rtde control script
rtde_c.stopRobot()

