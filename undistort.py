import numpy as np
import cv2
import math
from dataclasses import dataclass


@dataclass
class CameraSettings:
    def __init__(self, imageSize: tuple, diagonalMm: float, focalLenMm: float):
        self.imageSize = imageSize
        self.diagonalMm = diagonalMm
        self.focalLenMm = focalLenMm


CAM_SETTINGS = CameraSettings(
    (3840, 2160), 
    16.0 / 2.8,  # ≈5.714 mm (vidicon standard conversion))
    2.8
)

# k1, k2, p1, p2, k3
CAM_DISTORT_COEFFS = [
    -0.276, 0.06, 0.0084, -0.0016, -0.0044
]

class DistortionCorrector:
    def __init__(self, cameraSettings: CameraSettings, distortionCoeffs: list):
        self.width = cameraSettings.imageSize[0]
        self.height = cameraSettings.imageSize[1]
        self.diagonal_mm = cameraSettings.diagonalMm
        self.focal_length_mm = cameraSettings.focalLenMm
        self.dist = np.array(distortionCoeffs, dtype=np.float32)
        self.K = self.calculate_camera_matrix()
        print("cameraMatrix:", self.K)
        self.map1, self.map2, self.roi = self.create_undistort_maps()

    def calculate_camera_matrix(self):
        """Calculate camera intrinsic matrix"""
        aspect_ratio = self.width / self.height
        height_mm = self.diagonal_mm / math.sqrt(aspect_ratio**2 + 1)
        width_mm = aspect_ratio * height_mm
        
        fx = (self.focal_length_mm * self.width) / width_mm
        fy = (self.focal_length_mm * self.height) / height_mm
        
        return np.array([
            [fx, 0, self.width/2],
            [0, fy, self.height/2],
            [0, 0, 1]
        ], dtype=np.float32)

    def create_undistort_maps(self):
        """Generate undistortion maps with current distortion coefficients"""
        # Get optimal camera matrix and ROI for cropping
        new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, (self.width, self.height), 0, (self.width, self.height)
        )
        
        # Generate maps
        map1, map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, new_cam_matrix, (self.width, self.height), cv2.CV_32FC1
        )
        
        return map1, map2, roi
        
    def undistort_frame(self, frame):
        # Apply distortion correction
        undistorted = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        # Crop to region of interest (ROI) for maximum valid area
        x, y, w, h = self.roi
        undistorted = undistorted[y:y+h, x:x+w]
       
        return undistorted
 
# distCorrector = DistortionCorrector(CAM_SETTINGS, CAM_DISTORT_COEFFS)
# undistorted = distCorrector.undistort_frame(sourceFrame)
