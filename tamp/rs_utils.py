import numpy as np
import pyrealsense2 as rs

def rs_intrinsics_to_opencv_intrinsics(intr):
    D = np.array(intr.coeffs)
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]])
    return K, D

def get_serial_number(pipeline_profile):
    return pipeline_profile.get_device().get_info(rs.camera_info.serial_number)

def get_intrinsics(pipeline_profile, stream=rs.stream.color):
    stream_profile = pipeline_profile.get_stream(stream) # Fetch stream profile for depth stream
    intr = stream_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    return rs_intrinsics_to_opencv_intrinsics(intr)

class CaptureRS:
    def __init__(
        self, callback=None, vis=False, serial_number=None, intrinsics=None, min_tags=1
    ):
        self.callback = callback
        self.vis = vis
        self.min_tags = min_tags

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_number is not None:
            config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)

        # Start streaming
        pipeline_profile = self.pipeline.start(config)

        # And get the device info
        self.serial_number = get_serial_number(pipeline_profile)
        print(f"Connected to {self.serial_number}")

        # get the camera intrinsics
        # print('Using default intrinsics from camera.')
        if intrinsics is None:
            self.intrinsics = get_intrinsics(pipeline_profile)
        else:
            self.intrinsics = intrinsics

    def capture(self):
        for _ in range(100):
            # Wait for a coherent pair of frames: depth and color
            frameset = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(frameset.get_color_frame().get_data())

        return color_image, depth_image

    def close(self):
        self.pipeline.stop()
