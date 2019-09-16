
function getTransformStamped(objHandle,name,relTo,relToName)
    t=sim.getSystemTime()
    p=sim.getObjectPosition(objHandle,relTo)
    o=sim.getObjectQuaternion(objHandle,relTo)
    return {
        header={
            stamp=t,
            frame_id=relToName
        },
        child_frame_id=name,
        transform={
            translation={x=p[1],y=p[2],z=p[3]},
            rotation={x=o[1],y=o[2],z=o[3],w=o[4]}
        }
    }
end

function subscriber_callback(msg)
    -- This is the subscriber callback function
    sim.addStatusbarMessage('subscriber receives following Float32: '..msg.data)
end

function sysCall_threadmain()


    -- The child script initialization
    objectHandle=sim.getObjectAssociatedWithScript(sim.handle_self)
    objectName=sim.getObjectName(objectHandle)
    -- Check if the required RosInterface is there:
    moduleName=0
    index=0
    rosInterfacePresent=false
    while moduleName do
        moduleName=sim.getModuleName(index)
        if (moduleName=='RosInterface') then
            rosInterfacePresent=true

        end
        index=index+1
    end

    -- Prepare the float32 publisher and subscriber (we subscribe to the topic we advertise):
    if rosInterfacePresent then
        publisher=simROS.advertise('/simulationTime','std_msgs/Float32')
        subscriber=simROS.subscribe('/simulationTime','std_msgs/Float32','subscriber_callback')
    end


    robotHandle=sim.getObjectAssociatedWithScript(sim.handle_self)
    jointHandles={-1,-1,-1,-1,-1,-1}
    for i=1,6,1 do
        jointHandles[i]=sim.getObjectHandle('UR5_joint'..i)
    end

    -- Set-up some of the RML vectors:
    vel=180
    accel=40
    jerk=80
    currentVel={0,0,0,0,0,0,0}
    currentAccel={0,0,0,0,0,0,0}
    maxVel={vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180}
    maxAccel={accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180}
    maxJerk={jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180}
    targetVel={0,0,0,0,0,0}


    while sim.getSimulationState()~=sim.simulation_advancing_abouttostop do
        targetPos1={90*math.pi/180,90*math.pi/180,-90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180}
        sim.rmlMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos1,targetVel)
        
        simROS.sendTransform(getTransformStamped(sim.getObjectHandle('UR5_joint6'),'UR5_BaseEnd1',sim.getObjectHandle('UR5_joint1'),'UR5_joint1'))

        targetPos2={-90*math.pi/180,45*math.pi/180,90*math.pi/180,135*math.pi/180,90*math.pi/180,90*math.pi/180}
        sim.rmlMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos2,targetVel)
        
        simROS.sendTransform(getTransformStamped(sim.getObjectHandle('UR5_joint6'),'UR5_BaseEnd2',sim.getObjectHandle('UR5_joint1'),'UR5_joint1'))

        targetPos3={0,0,0,0,0,0}
        sim.rmlMoveToJointPositions(jointHandles,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos3,targetVel)
        
        simROS.sendTransform(getTransformStamped(sim.getObjectHandle('UR5_joint6'),'UR5_BaseEnd3',sim.getObjectHandle('UR5_link1_visible'),'UR5_joint1'))

        if rosInterfacePresent then
            simROS.publish(publisher,{data=sim.getSimulationTime()})
            -- To send several transforms at once, use simROS.sendTransforms instead
        end
    end
end


function sysCall_cleanup()
	-- Put some clean-up code here:
    
    if rosInterfacePresent then
        simROS.shutdownPublisher(publisher)
        simROS.shutdownSubscriber(subscriber)
    end

end

