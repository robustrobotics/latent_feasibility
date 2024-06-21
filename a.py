import pybullet as p
import pybullet_data
import time

# Connect to the PyBullet physics server
physicsClient = p.connect(p.GUI)  # Use p.DIRECT for non-graphical version

# Set the search path to the pybullet_data package
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a plane and a simple robot (a sphere)
planeId = p.loadURDF("plane.urdf")
sphereId = p.loadURDF("sphere2.urdf", [0, 0, 1])  # Position sphere above the plane

# Set gravity
p.setGravity(0, 0, -9.81)

# Run the simulation for 240 steps (about 4 seconds)
for i in range(240):
    p.stepSimulation()
    time.sleep(1./60.)

# Disconnect from the server
p.disconnect()

print("PyBullet simulation ran successfully!")
