import time

import numpy as np
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray

try:
    import CoppeliaSim.sim as sim
except:
    print('--------------------------------------------------------------')
    print('"sim.py" could not be imported. This means very probably that')
    print('either "sim.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "sim.py"')
    print('--------------------------------------------------------------')
    print('')

import time


class CoppeliaSim(object):

    def __init__(self, robotStr):
        # Init ROS node
        rospy.init_node('CoppeliaSim', anonymous=False)

        self.robotNumber = robotStr

        # publishers
        self.startSimPublisher = rospy.Publisher('/startSimulation', Bool, queue_size=5)
        self.stopSimPublisher = rospy.Publisher('/stopSimulation', Bool, queue_size=5)
        self.gripperSimPublisher = rospy.Publisher('/closeGripper' + self.robotNumber, Bool, queue_size=5)
        self.robotQPublisher = rospy.Publisher('/robotQ' + self.robotNumber, Float32MultiArray, queue_size=5)

        while self.startSimPublisher.get_num_connections() == 0 or self.stopSimPublisher.get_num_connections() == 0:
            rospy.loginfo("Waiting for subscriber to connect")
            rospy.sleep(1)

        # subscribers
        self.isMovingSubscriber = rospy.Subscriber('/robotMoving' + self.robotNumber, Bool, self.robotMovingCallback)
        self.simStateSubscriber = rospy.Subscriber('/simulationState', Int32, self.stateCallback)

        # Simulation parameters
        self.simState = 0
        self.isMoving = False
        self.defaultQ = np.array([0, -1.57, 0, -1.57, 0, 0], dtype=np.float32)
        self.currentQ = self.defaultQ

        #init remoteAPI
        self.remoteClientID = -1

    def __init__(self, robotStr, clientID):
        # Init ROS node
        rospy.init_node('CoppeliaSim', anonymous=False)

        self.robotNumber = robotStr

        # publishers
        self.startSimPublisher = rospy.Publisher('/startSimulation', Bool, queue_size=5)
        self.stopSimPublisher = rospy.Publisher('/stopSimulation', Bool, queue_size=5)
        self.gripperSimPublisher = rospy.Publisher('/closeGripper' + self.robotNumber, Bool, queue_size=5)
        self.robotQPublisher = rospy.Publisher('/robotQ' + self.robotNumber, Float32MultiArray, queue_size=5)

        while self.startSimPublisher.get_num_connections() == 0 or self.stopSimPublisher.get_num_connections() == 0:
            rospy.loginfo("Waiting for subscriber to connect")
            rospy.sleep(1)

        # subscribers
        self.isMovingSubscriber = rospy.Subscriber('/robotMoving' + self.robotNumber, Bool, self.robotMovingCallback)
        self.simStateSubscriber = rospy.Subscriber('/simulationState', Int32, self.stateCallback)

        # Simulation parameters
        self.simState = 0
        self.isMoving = False
        self.defaultQ = np.array([0, -1.57, 0, -1.57, 0, 0], dtype=np.float32)
        self.currentQ = self.defaultQ

        #init remoteAPI
        self.remoteClientID = clientID

    def __del__(self):
        rospy.loginfo("[CoppeliaSim] Destructor called, initiating cleanup")
        self.startSimPublisher.unregister()
        self.stopSimPublisher.unregister()
        self.gripperSimPublisher.unregister()
        self.robotQPublisher.unregister()

        self.isMovingSubscriber.unregister()
        self.simStateSubscriber.unregister()

        # Shut down remote api
        # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        # sim.simxGetPingTime(self.remoteClientID)
        # sim.simxFinish(self.remoteClientID)

    def remoteApiTest(self):

        # Now try to retrieve data in a blocking fashion (i.e. a service call):
        res, objs = sim.simxGetObjects(self.remoteClientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print('Number of objects in the scene: ', len(objs))
        else:
            print('Remote API function call returned with error code: ', res)

    """ 
    Attributes: 
        std_msgs::Int32 msg
    """

    def stateCallback(self, msg):
        self.simState = msg.data

    """
    Attributes: 
        std_msgs::Bool
    """

    def robotMovingCallback(self, msg):
        self.isMoving = msg.data

    """ 
    Attributes: 
        robot config containing 
    """

    def setQ(self, config):

        while self.robotQPublisher.get_num_connections() == 0:
            rospy.loginfo("Waiting for subscriber to connect")
            rospy.sleep(1)

        # init clock for timeout
        startTime = time.time()

        while self.robotMoving() == True:
            # 20 second timeout to reach goal
            if (time.time() - startTime > 20):
                return False
            time.sleep(0.1)

        self.publishQ(config)

    """ 
    Moves the robot to the homing position
    """

    def moveQ(self):
        pass

    def moveHome(self):
        self.setQ(self.defaultQ)
        pass

    def publishQ(self, q):
        msg = Float32MultiArray()
        # subtract the default Q to make the Q correspond to the real Q of UR robot
        msg.data = np.subtract(q, self.defaultQ)
        self.robotQPublisher.publish(msg)

    def simStopped(self):
        if self.simState == 0:
            return True
        else:
            return False

    def simRunning(self):
        if self.simState == 1:
            return True
        else:
            return False

    def startSim(self):
        rospy.loginfo("Starting simulation...")

        while not self.simRunning():
            self.startSimPublisher.publish(Bool())

        rospy.loginfo("Simulation started...")

    def stopSim(self):
        rospy.loginfo("Stopping simulation...")

        while not self.simStopped():
            self.stopSimPublisher.publish(Bool())

        rospy.loginfo("Simulation stopped...")

    def closeGripper(self):

        while self.gripperSimPublisher.get_num_connections() == 0 == 0:
            rospy.loginfo("Waiting for subscriber to connect")
            rospy.sleep(1)

        msg = Bool()
        msg.data = True
        self.gripperSimPublisher.publish(msg)

    def openGripper(self):

        while self.gripperSimPublisher.get_num_connections() == 0 == 0:
            rospy.loginfo("Waiting for subscriber to connect")
            rospy.sleep(1)

        msg = Bool()
        msg.data = False
        self.gripperSimPublisher.publish(msg)

    def robotMoving(self):
        rospy.wait_for_message(topic='/robotMoving' + self.robotNumber, topic_type=Bool)
        return self.isMoving

    def getPosition(self, entityString):
        objectHandle = sim.simxGetObjectHandle(clientID=self.remoteClientID, objectName=entityString, operationMode=sim.simx_opmode_blocking)
        return sim.simxGetObjectPosition(clientID=self.remoteClientID, objectHandle=objectHandle, operationMode=sim.simx_opmode_blocking)

    def getJointAngles(self, robotString):
        jointHandle = sim.simxGetObjectHandle(clientID=self.remoteClientID, objectName=robotString, operationMode=sim.simx_opmode_blocking)
        return sim.simxGetJointPosition(clientID=self.remoteClientID, jointHandle=jointHandle, operationMode=sim.simx_opmode_blocking)


    def getOrientation(self, entityString):
        objectHandle = sim.simxGetObjectHandle(clientID=self.remoteClientID, objectName=entityString, operationMode=sim.simx_opmode_blocking)
        return sim.simxGetObjectOrientation(clientID=self.remoteClientID, objectHandle=objectHandle, operationMode=sim.simx_opmode_blocking)

    def startSimRemoteAPI(self):
        pass

    def stopSimRemoteAPI(self):
        pass

    def resetSim(self):
        pass
