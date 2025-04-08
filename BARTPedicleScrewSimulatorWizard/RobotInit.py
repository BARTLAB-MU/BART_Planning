import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

logger = logging.getLogger(__name__)

def mc_mean(transforms, resolution):
    """Calculate mean of transformation matrices"""
    try:
        sum_matrix = np.zeros_like(transforms[:, :, 0])
        for i in range(resolution):
            sum_matrix += transforms[:, :, i]
        return sum_matrix / resolution
    except Exception as e:
        logger.error(f"Error in mc_mean: {str(e)}")
        return np.eye(4)

def m_correction(matrix):
    """Apply orthogonalization correction to rotation matrix"""
    try:
        # Use SVD for orthogonalization (Python equivalent of MATLAB's M*(inv(M'*M)^0.5))
        u, s, vh = np.linalg.svd(matrix, full_matrices=True)
        return u @ vh
    except Exception as e:
        logger.error(f"Error in m_correction: {str(e)}")
        return np.eye(3)

class Robot:
    def __init__(self, resolution, reach):
        """
        Initialize a turret-like robot with:
        - Link 1: Base that only rotates around Y axis (elevation)
        - Link 2: Barrel that rotates around Z axis (azimuth) and translates along Y axis
        
        Parameters:
            resolution: Number of samples in joint space
            reach: Maximum reach distance
        """
        self.resolution = resolution
        self.reach = reach
        self.joint_1_limit = [-180, 180]
        self.joint_2_limit = [-90, 90]
        
        # Generate joint space
        self.link_1 = self._create_link1_transforms()
        self.link_2 = self._create_link2_transforms()
        
        # Calculate mean transforms
        self.h1_mean, self.h2_mean = self.get_mean_transforms()
    
    def _create_link1_transforms(self):
        """Create Link 1 transforms (rotation around Y only)"""
        link_1 = np.zeros((4, 4, self.resolution))
        joint_angles = np.linspace(self.joint_1_limit[0], self.joint_1_limit[1], self.resolution)
        
        for i, angle in enumerate(joint_angles):
            # Create rotation matrix around Y axis
            rot_matrix = R.from_euler('y', angle, degrees=True).as_matrix()
            
            # Create 4x4 transform matrix (rotation only)
            transform = np.eye(4)
            transform[0:3, 0:3] = rot_matrix
            link_1[:, :, i] = transform
        
        return link_1
    
    def _create_link2_transforms(self):
        """Create Link 2 transforms (rotation around Z + translation along Y)"""
        link_2 = np.zeros((4, 4, self.resolution))
        joint_angles = np.linspace(self.joint_2_limit[0], self.joint_2_limit[1], self.resolution)
        
        for i, angle in enumerate(joint_angles):
            # Create rotation matrix around Z axis
            rot_matrix = R.from_euler('z', angle, degrees=True).as_matrix()

            rot_transform = np.eye(4)
            rot_transform[0:3,0:3] = rot_matrix

            trans_transform = np.eye(4)
            trans_transform[0,3] = self.reach * np.cos(np.deg2rad(angle))
            trans_transform[1,3] = self.reach * np.sin(np.deg2rad(angle))
            
            # Create transform with rotation and translation
            transform = rot_transform @ trans_transform
            link_2[:, :, i] = transform
        
        return link_2
    
    def get_mean_transforms(self):
        """Calculate mean transforms for both links"""
        # Mean of Link 1 rotations
        h1_mean_rot = mc_mean(self.link_1[0:3, 0:3, :], self.resolution)
        h1_mean_rot = m_correction(h1_mean_rot)
        
        # Mean of Link 1 translations
        h1_mean_trans = mc_mean(self.link_1, self.resolution)
        
        # Combine into transform
        h1_mean = np.eye(4)
        h1_mean[0:3, 0:3] = h1_mean_rot
        h1_mean[0:3, 3] = h1_mean_trans[0:3, 3]
        
        # Mean of Link 2 rotations
        h2_mean_rot = mc_mean(self.link_2[0:3, 0:3, :], self.resolution)
        h2_mean_rot = m_correction(h2_mean_rot)
        
        # Mean of Link 2 translations
        h2_mean_trans = mc_mean(self.link_2, self.resolution)
        
        # Combine into transform
        h2_mean = np.eye(4)
        h2_mean[0:3, 0:3] = h2_mean_rot
        h2_mean[0:3, 3] = h2_mean_trans[0:3, 3]
        
        return h1_mean, h2_mean