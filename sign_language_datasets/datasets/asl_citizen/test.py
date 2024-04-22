from pose_format import Pose

data_buffer = open("5983320426555474-LICENSE.pose", "rb").read()
pose = Pose.read(data_buffer)

numpy_data = pose.body.fps
confidence_measure  = pose.body.confidence
print("Shape of Pose Data:", numpy_data)