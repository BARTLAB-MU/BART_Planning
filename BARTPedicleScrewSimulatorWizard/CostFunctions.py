import numpy as np
import vtk
import logging
import slicer

logger = logging.getLogger(__name__)

def bresenham_3d(origin, endpoint, matrix):
    """
    Draw a 3D line between origin and endpoint in the voxel matrix using Bresenham's algorithm.
    Returns a binary mask where line voxels are 1 and others are 0.
    
    Parameters:
        origin (array): [x, y, z] start point
        endpoint (array): [x, y, z] end point
        matrix (array): 3D numpy array representing the volume
        
    Returns:
        array: Binary mask with same dimensions as matrix
    """
    try:
        # Get matrix dimensions
        matrix_size = matrix.shape
        
        # Round endpoints to integers
        x1, y1, z1 = map(int, np.round(origin))
        x2, y2, z2 = map(int, np.round(endpoint))
        
        # Calculate differences and signs
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        sz = 1 if z2 > z1 else -1
        
        # Determine dominant axis
        if dx >= dy and dx >= dz:
            err_y = 2 * dy - dx
            err_z = 2 * dz - dx
            nsteps = dx
        elif dy >= dx and dy >= dz:
            err_x = 2 * dx - dy
            err_z = 2 * dz - dy
            nsteps = dy
        else:
            err_x = 2 * dx - dz
            err_y = 2 * dy - dz
            nsteps = dz
        
        # Create result matrix
        result = np.zeros(matrix_size, dtype=np.uint8)
        
        # Draw line using Bresenham's algorithm
        x, y, z = x1, y1, z1
        
        for i in range(nsteps + 1):
            # Check if point is within bounds
            if 0 <= x < matrix_size[0] and 0 <= y < matrix_size[1] and 0 <= z < matrix_size[2]:
                result[x, y, z] = 1
                
            # Update coordinates based on dominant axis
            if dx >= dy and dx >= dz:
                if err_y >= 0:
                    y += sy
                    err_y -= 2 * dx
                if err_z >= 0:
                    z += sz
                    err_z -= 2 * dx
                err_y += 2 * dy
                err_z += 2 * dz
                x += sx
            elif dy >= dx and dy >= dz:
                if err_x >= 0:
                    x += sx
                    err_x -= 2 * dy
                if err_z >= 0:
                    z += sz
                    err_z -= 2 * dy
                err_x += 2 * dx
                err_z += 2 * dz
                y += sy
            else:
                if err_x >= 0:
                    x += sx
                    err_x -= 2 * dz
                if err_y >= 0:
                    y += sy
                    err_y -= 2 * dz
                err_x += 2 * dx
                err_y += 2 * dy
                z += sz
        
        return result
        
    except Exception as e:
        logger.error(f"Error in bresenham_3d: {str(e)}")
        # Return empty result on error
        return np.zeros(matrix.shape, dtype=np.uint8)

def find_closest_point_to_line(pt_cloud, line_origin, line_direction):
    """
    Find the closest point in pt_cloud to the line defined by origin and direction.
    
    Parameters:
        pt_cloud (vtkPolyData): Point cloud to search
        line_origin (array): [x, y, z] origin of the line
        line_direction (array): Direction vector of the line
        
    Returns:
        tuple: (closest_point, min_distance)
    """
    try:
        # Normalize direction vector
        direction_mag = np.linalg.norm(line_direction)
        if direction_mag < 1e-6:
            logger.warning("Line direction vector is too short")
            return line_origin, float('inf')
            
        line_direction = line_direction / direction_mag
        
        # Validate point cloud
        if not pt_cloud or not hasattr(pt_cloud, 'GetPoints'):
            logger.warning("Invalid point cloud object")
            return line_origin, float('inf')
            
        points_vtk = pt_cloud.GetPoints()
        if not points_vtk or points_vtk.GetNumberOfPoints() == 0:
            logger.warning("Empty point cloud or no points found")
            return line_origin, float('inf')
            
        # Convert VTK points to numpy array
        n_points = points_vtk.GetNumberOfPoints()
        points = np.zeros((n_points, 3))
        for i in range(n_points):
            points[i] = points_vtk.GetPoint(i)
        
        # Compute vectors from line origin to each point
        vec_to_points = points - line_origin
        
        # Project vectors onto line direction
        proj_scalars = np.sum(vec_to_points * line_direction, axis=1)
        
        # Compute closest points on line
        closest_points = line_origin + proj_scalars[:, np.newaxis] * line_direction
        
        # Compute distances
        distances = np.linalg.norm(points - closest_points, axis=1)
        
        # Find minimum distance and corresponding point
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        closest_point = points[min_idx]
        
        logger.debug(f"Closest point found at distance: {min_distance}")
        return closest_point, min_distance
        
    except Exception as e:
        logger.error(f"Error in find_closest_point_to_line: {str(e)}")
        return line_origin, float('inf')

def point_to_line_distance(point, line_origin, line_direction):
    """
    Calculate the perpendicular distance from a point to a line.
    This function includes detailed debugging to trace the calculation.
    
    Parameters:
        point (array): [x, y, z] point to measure from
        line_origin (array): [x, y, z] origin of the line
        line_direction (array): Direction vector of the line
        
    Returns:
        float: Distance from point to line
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs and convert to numpy arrays
        if point is None or line_origin is None or line_direction is None:
            logger.warning("Invalid inputs to point_to_line_distance")
            return float('inf')
            
        point = np.array(point, dtype=np.float64)
        line_origin = np.array(line_origin, dtype=np.float64)
        line_direction = np.array(line_direction, dtype=np.float64)
        
        # Debug inputs
        logger.debug(f"point_to_line_distance inputs:")
        logger.debug(f"  point: {point}")
        logger.debug(f"  line_origin: {line_origin}")
        logger.debug(f"  line_direction: {line_direction}")
        
        # Normalize direction vector
        direction_mag = np.linalg.norm(line_direction)
        if direction_mag < 1e-6:
            logger.warning("Line direction vector is too short")
            return float('inf')
            
        normalized_direction = line_direction / direction_mag
        logger.debug(f"  normalized_direction: {normalized_direction}")
        
        # Vector from line origin to the point
        vec_to_point = point - line_origin
        logger.debug(f"  vec_to_point: {vec_to_point}")
        
        # Method 1: Cross product
        # The cross product gives a vector perpendicular to both input vectors,
        # with magnitude equal to the area of the parallelogram they form.
        # The perpendicular distance is this area divided by the magnitude of the line direction.
        cross_product = np.cross(vec_to_point, normalized_direction)
        distance1 = np.linalg.norm(cross_product)
        logger.debug(f"  cross_product: {cross_product}")
        logger.debug(f"  distance (cross product method): {distance1}")
        
        # Method 2: Projection
        # Project the vector onto the line direction, then use Pythagoras to find the perpendicular component
        dot_product = np.dot(vec_to_point, normalized_direction)
        projection = normalized_direction * dot_product
        perpendicular_vector = vec_to_point - projection
        distance2 = np.linalg.norm(perpendicular_vector)
        logger.debug(f"  dot_product: {dot_product}")
        logger.debug(f"  projection: {projection}")
        logger.debug(f"  perpendicular_vector: {perpendicular_vector}")
        logger.debug(f"  distance (projection method): {distance2}")
        
        # Method 3: Formula d = |v × w| / |w|
        # Where v is vec_to_point and w is normalized_direction
        # Since w is already normalized, this simplifies to d = |v × w|
        cross_magnitude = np.linalg.norm(np.cross(vec_to_point, normalized_direction))
        distance3 = cross_magnitude
        logger.debug(f"  distance (formula method): {distance3}")
        
        # The results from all methods should be identical (within floating point precision)
        # Return the result from Method 2 (most explicit)
        return distance2
        
    except Exception as e:
        logger.error(f"Error in point_to_line_distance: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return float('inf')

def gen_traj(base, ee):
    """
    Generate trajectory vector from base to end-effector.
    
    Parameters:
        base: 4x4 transformation matrix for base
        ee: 4x4 transformation matrix for end-effector
        
    Returns:
        array: Normalized direction vector
    """
    # Extract translation components
    base_coord = base[0:3, 3]
    ee_coord = ee[0:3, 3]
    
    # Calculate direction vector
    vec = ee_coord - base_coord
    
    # Normalize
    return vec / np.linalg.norm(vec)

def sample_along_trajectory(volume_node, mask_node, start_point, direction, length, num_samples=50):
    """
    Sample CT values and mask values along a trajectory using Bresenham's algorithm.
    
    Parameters:
        volume_node (vtkMRMLScalarVolumeNode): CT volume node
        mask_node (vtkMRMLLabelMapVolumeNode): Segmentation mask
        start_point (array): Starting point [x, y, z] in RAS coordinates
        direction (array): Direction vector (normalized)
        length (float): Length of trajectory
        num_samples (int): Number of samples to take
        
    Returns:
        tuple: (ct_samples, mask_samples) Arrays of CT and mask values
    """
    import numpy as np
    import vtk
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Convert to numpy arrays if not already
        start_point = np.array(start_point)
        direction = np.array(direction) / np.linalg.norm(direction)  # Ensure normalized
        
        # Calculate end point
        end_point = start_point + direction * length
        
        # Get RAS to IJK matrices
        volume_ras_to_ijk = vtk.vtkMatrix4x4()
        mask_ras_to_ijk = vtk.vtkMatrix4x4()
        
        volume_node.GetRASToIJKMatrix(volume_ras_to_ijk)
        mask_node.GetRASToIJKMatrix(mask_ras_to_ijk)
        
        # Convert start and end points to IJK coordinates
        start_ijk = np.zeros(4)
        end_ijk = np.zeros(4)
        
        volume_ras_to_ijk.MultiplyPoint(np.append(start_point, 1.0), start_ijk)
        volume_ras_to_ijk.MultiplyPoint(np.append(end_point, 1.0), end_ijk)
        
        # Round to integers for Bresenham
        start_ijk = np.round(start_ijk[:3]).astype(int)
        end_ijk = np.round(end_ijk[:3]).astype(int)
        
        # Get volume dimensions
        volume_image = volume_node.GetImageData()
        volume_dims = volume_image.GetDimensions()
        
        # Get mask dimensions
        mask_image = mask_node.GetImageData()
        mask_dims = mask_image.GetDimensions()
        
        # Initialize points list
        points = []
        
        # Add the starting point
        points.append((start_ijk[0], start_ijk[1], start_ijk[2]))
        
        # Use Bresenham's 3D line algorithm
        # Based on https://www.geeksforgeeks.org/bresenhams-algorithm-for-3d-line-drawing/
        x1, y1, z1 = start_ijk
        x2, y2, z2 = end_ijk
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1
        
        # Choosing the driving axis
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                points.append((x1, y1, z1))
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                points.append((x1, y1, z1))
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                points.append((x1, y1, z1))
        
        # Sample at evenly spaced points along the line
        if len(points) > num_samples:
            # Subsample if we have more points than needed
            indices = np.linspace(0, len(points)-1, num_samples, dtype=int)
            points = [points[i] for i in indices]
        
        # Initialize arrays for CT and mask values
        ct_samples = []
        mask_samples = []
        
        # Sample values at each point
        for x, y, z in points:
            # Check if point is within CT volume bounds
            if (0 <= x < volume_dims[0] and 0 <= y < volume_dims[1] and 0 <= z < volume_dims[2]):
                # Get CT value
                ct_value = volume_image.GetScalarComponentAsDouble(x, y, z, 0)
                ct_samples.append(ct_value)
                
                # Convert to mask IJK coordinates if needed
                mask_x, mask_y, mask_z = x, y, z
                if mask_ras_to_ijk != volume_ras_to_ijk:
                    # If mask has different IJK space, convert RAS to mask IJK
                    ras_point = np.zeros(4)
                    ijk_to_ras = vtk.vtkMatrix4x4()
                    volume_node.GetIJKToRASMatrix(ijk_to_ras)
                    ijk_to_ras.MultiplyPoint([x, y, z, 1], ras_point)
                    
                    mask_ijk = np.zeros(4)
                    mask_ras_to_ijk.MultiplyPoint(ras_point, mask_ijk)
                    mask_x, mask_y, mask_z = np.round(mask_ijk[:3]).astype(int)
                
                # Check if point is within mask volume bounds
                if (0 <= mask_x < mask_dims[0] and 0 <= mask_y < mask_dims[1] and 0 <= mask_z < mask_dims[2]):
                    # Get mask value (0 = outside, >0 = inside)
                    mask_value = mask_image.GetScalarComponentAsDouble(mask_x, mask_y, mask_z, 0)
                    mask_samples.append(mask_value)
                else:
                    # If outside mask bounds, treat as outside segmentation
                    mask_samples.append(0)
            else:
                # If outside CT bounds, use defaults
                ct_samples.append(0)
                mask_samples.append(0)
        
        return np.array(ct_samples), np.array(mask_samples)
        
    except Exception as e:
        logger.error(f"Error in sample_along_trajectory: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return np.array([]), np.array([])

def compute_safety_margin(trajectory_points, vertebra_model):
    """
    Compute the safety margin between trajectory and vertebra surface.
    
    Parameters:
        trajectory_points (array): Array of points along trajectory
        vertebra_model (vtkPolyData): Surface model of vertebra
        
    Returns:
        float: Minimum distance from trajectory to surface
    """
    try:
        if not vertebra_model or not hasattr(vertebra_model, 'GetPoints'):
            logger.warning("Invalid vertebra model")
            return 0.0
            
        # Create a locator for fast distance queries
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(vertebra_model)
        locator.BuildLocator()
        
        # Find minimum distance from trajectory to surface
        min_distance = float('inf')
        
        for point in trajectory_points:
            # Find closest point
            id = locator.FindClosestPoint(point)
            if id >= 0:
                surface_point = vertebra_model.GetPoint(id)
                distance = np.linalg.norm(np.array(point) - np.array(surface_point))
                min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            min_distance = 0.0
            
        return min_distance
        
    except Exception as e:
        logger.error(f"Error in compute_safety_margin: {str(e)}")
        return 0.0
    
def distance_cost(insertion_point, trajectory_direction, pedicle_center):
    """
    Calculate the perpendicular distance from the trajectory line to the pedicle center.
    This is a critical cost component that helps place the trajectory near the center of the pedicle.
    
    Parameters:
        insertion_point (array): 3D coordinates of insertion point
        trajectory_direction (array): Unit vector of trajectory direction
        pedicle_center (array): 3D coordinates of pedicle center
        
    Returns:
        float: Distance cost value
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs and convert to numpy arrays
        if insertion_point is None or trajectory_direction is None or pedicle_center is None:
            logger.warning("Invalid inputs to distance_cost")
            return float('inf')
            
        insertion_point = np.array(insertion_point, dtype=np.float64)
        trajectory_direction = np.array(trajectory_direction, dtype=np.float64)
        pedicle_center = np.array(pedicle_center, dtype=np.float64)
        
        # Normalize direction vector
        direction_mag = np.linalg.norm(trajectory_direction)
        if direction_mag < 1e-6:
            logger.warning("Trajectory direction vector is too short")
            return float('inf')
            
        normalized_direction = trajectory_direction / direction_mag
        
        # Vector from insertion point to pedicle center
        vec_to_center = pedicle_center - insertion_point
        
        # Project this vector onto the trajectory direction
        dot_product = np.dot(vec_to_center, normalized_direction)
        projection = normalized_direction * dot_product
        
        # The perpendicular component is the difference
        perpendicular_vector = vec_to_center - projection
        
        # The distance is the magnitude of this perpendicular component
        distance = np.linalg.norm(perpendicular_vector)
        
        # Calculate the closest point on the trajectory to the pedicle center
        closest_point = insertion_point + projection
        
        # If the dot product is negative, the projection is behind the insertion point
        # (pedicle center is behind trajectory origin)
        if dot_product < 0:
            logger.debug("Pedicle center is behind trajectory origin")
            
        # If distance is greater than a threshold, apply a penalty
        # This helps avoid positions where the trajectory is very far from the pedicle
        threshold = 20.0  # mm
        if distance > threshold:
            # Apply a quadratic penalty for distances beyond the threshold
            penalty = 1.0 + ((distance - threshold) / threshold) ** 2
            distance *= penalty
            logger.debug(f"Applied distance penalty: {penalty}, adjusted distance: {distance}")
        
        return distance
        
    except Exception as e:
        logger.error(f"Error in distance_cost: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return float('inf')
    
def bone_density_cost(insertion_point, trajectory_direction, volume_node, mask_node, trajectory_length, num_samples=50):
    """
    Calculate cost based on bone density along the trajectory.
    Higher density (cortical bone) is preferred for stronger fixation.
    
    Parameters:
        insertion_point (array): 3D coordinates of insertion point
        trajectory_direction (array): Unit vector of trajectory direction
        volume_node (vtkMRMLScalarVolumeNode): CT volume node
        mask_node (vtkMRMLLabelMapVolumeNode): Segmentation mask
        trajectory_length (float): Maximum length of trajectory
        num_samples (int): Number of points to sample along trajectory
        
    Returns:
        float: Density cost value (higher is better)
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Sample CT values and mask values along the trajectory using Bresenham's algorithm
        ct_samples, mask_samples = sample_along_trajectory(
            volume_node, 
            mask_node,
            insertion_point, 
            trajectory_direction, 
            trajectory_length, 
            num_samples
        )
        
        if len(ct_samples) == 0:
            logger.warning("No samples collected along trajectory")
            return -1000.0  # Large negative penalty for invalid trajectory
        
        # Calculate the average HU value along the trajectory
        # Higher HU values (cortical bone) are better for screw fixation
        density = np.sum(ct_samples)
        
        # Return the mean density directly as a positive cost factor
        # Higher density = higher return value = better trajectory
        return density
        
    except Exception as e:
        logger.error(f"Error in bone_density_cost: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return -1000.0  # Large negative penalty on error

def cost_total(
    insertion_point, 
    trajectory_direction,
    vertebra_model,
    pedicle_axis,
    pedicle_center,
    weights,
    volume_node,
    mask_node,
    trajectory_length
):
    """
    Calculate the total cost of a trajectory:
    1. Distance from pedicle center (minimize)
    2. Inverse of bone density along trajectory (minimize)
    
    Parameters:
        insertion_point: 3D coordinates of insertion point
        trajectory_direction: Unit vector of trajectory direction
        vertebra_model: Surface model of the vertebra
        pedicle_axis: Principal axis of the pedicle
        pedicle_center: 3D coordinates of pedicle center
        weights: Weights for different cost components [distance, density]
        volume_node: CT volume node
        mask_node: Segmentation mask
        trajectory_length: Maximum length of trajectory
        
    Returns:
        tuple: (total_cost, cost_components)
    """
    # Calculate distance cost (minimize)
    distance_value = distance_cost(insertion_point, trajectory_direction, pedicle_center)
    
    # Calculate density cost (maximize original value, so negate for minimization)
    density_value = bone_density_cost(insertion_point, trajectory_direction, volume_node, mask_node, trajectory_length)
    
    # Since we want to maximize the density but minimize the distance,
    # we need to invert the density cost
    if density_value > 0:
        # Normal case - invert density value to convert maximization to minimization
        density_cost = -density_value  # Negative so lower values (to minimize) = higher density
    else:
        # Penalty case (outside segmentation) - keep the negative value
        density_cost = density_value
    
    # Apply weights
    if weights is not None and len(weights) >= 2:
        total_cost = weights[0] * distance_value + weights[1] * density_cost
    else:
        total_cost = distance_value + density_cost
    
    cost_components = {"distance": distance_value, "density": density_cost}
    
    return total_cost, cost_components

def visualize_trajectory(insertion_point, trajectory_direction, pedicle_center, trajectory_length=100.0, name_prefix="Debug"):
    """
    Create visualization objects for a trajectory and its relationship to the pedicle center.
    
    Parameters:
        insertion_point (array): 3D coordinates of insertion point
        trajectory_direction (array): Unit vector of trajectory direction
        pedicle_center (array): 3D coordinates of pedicle center
        trajectory_length (float): Length of the trajectory line
        name_prefix (str): Prefix for node names
        
    Returns:
        tuple: (trajectory_node, distance_line_node, point_node) - MRML nodes for visualization
    """
    import slicer
    import vtk
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Calculate the trajectory end point
        end_point = insertion_point + trajectory_direction * trajectory_length
        
        # Create a line source for the trajectory
        trajectory_line = vtk.vtkLineSource()
        trajectory_line.SetPoint1(insertion_point)
        trajectory_line.SetPoint2(end_point)
        trajectory_line.Update()
        
        # Create a model node for the trajectory
        trajectory_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{name_prefix}_Trajectory")
        trajectory_node.SetAndObservePolyData(trajectory_line.GetOutput())
        
        # Create a display node for the trajectory
        trajectory_display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        trajectory_display.SetColor(0.0, 1.0, 0.0)  # Green
        trajectory_display.SetLineWidth(3.0)
        trajectory_node.SetAndObserveDisplayNodeID(trajectory_display.GetID())
        
        # Calculate the closest point on the trajectory to the pedicle center
        # Project pedicle_center onto the trajectory line
        vec_to_center = pedicle_center - insertion_point
        projection = np.dot(vec_to_center, trajectory_direction) * trajectory_direction
        closest_point = insertion_point + projection
        
        # Create a line source for the distance line
        distance_line = vtk.vtkLineSource()
        distance_line.SetPoint1(pedicle_center)
        distance_line.SetPoint2(closest_point)
        distance_line.Update()
        
        # Create a model node for the distance line
        distance_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{name_prefix}_Distance")
        distance_node.SetAndObservePolyData(distance_line.GetOutput())
        
        # Create a display node for the distance line
        distance_display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        distance_display.SetColor(1.0, 0.0, 0.0)  # Red
        distance_display.SetLineWidth(2.0)
        distance_node.SetAndObserveDisplayNodeID(distance_display.GetID())
        
        # Create a point for the pedicle center
        point_source = vtk.vtkSphereSource()
        point_source.SetCenter(pedicle_center)
        point_source.SetRadius(2.0)
        point_source.Update()
        
        # Create a model node for the point
        point_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{name_prefix}_Center")
        point_node.SetAndObservePolyData(point_source.GetOutput())
        
        # Create a display node for the point
        point_display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        point_display.SetColor(1.0, 1.0, 0.0)  # Yellow
        point_node.SetAndObserveDisplayNodeID(point_display.GetID())
        
        # Calculate and log the distance
        distance = np.linalg.norm(pedicle_center - closest_point)
        logger.info(f"Distance from trajectory to pedicle center: {distance:.2f} mm")
        
        return trajectory_node, distance_node, point_node
        
    except Exception as e:
        logger.error(f"Error in visualize_trajectory: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None


def visualize_search_result(vertebra, insertion_point, final_traj, angles, cost, name_prefix="Optimal"):
    """
    Create visualization of the final search result.
    
    Parameters:
        vertebra: Vertebra object containing anatomical information
        insertion_point: 3D coordinates of insertion point
        final_traj: Vector representing the optimal trajectory direction
        angles: Tuple of (vertical_angle, horizontal_angle) in degrees
        cost: Cost value of the final trajectory
        name_prefix: Prefix for node names
        
    Returns:
        tuple: (trajectory_node, distance_node, point_node, info_node) - MRML nodes for visualization
    """
    import slicer
    import vtk
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create basic trajectory visualization
        trajectory_node, distance_node, point_node = visualize_trajectory(
            insertion_point, 
            final_traj, 
            vertebra.pedicle_center_point if hasattr(vertebra, 'pedicle_center_point') else vertebra.centroid, 
            trajectory_length=100.0, 
            name_prefix=name_prefix
        )
        
        # Create a text annotation for the result
        vertical_angle, horizontal_angle = angles
        info_text = f"Trajectory Info:\n"
        info_text += f"Vertical Angle: {vertical_angle:.1f}°\n"
        info_text += f"Horizontal Angle: {horizontal_angle:.1f}°\n"
        info_text += f"Cost: {cost:.2f}\n"
        
        info_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", f"{name_prefix}_Info")
        info_node.SetText(info_text)
        
        # Log the result
        logger.info(f"\n{info_text}")
        
        return trajectory_node, distance_node, point_node, info_node
        
    except Exception as e:
        logger.error(f"Error in visualize_search_result: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None
    
def test_distance_cost_function():
    """
    Test and validate the distance cost function with controlled inputs.
    Creates a simple test case and visualizes the results.
    
    This function can be called from the Python console in Slicer to debug the cost function:
        from BARTPedicleScrewSimulatorWizard.CostFunctions import test_distance_cost_function
        test_distance_cost_function()
    """
    import slicer
    import vtk
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed output
    
    # Create console handler if not already present
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    logger.info("Running distance cost function test...")
    
    # Create test case with known geometry
    # Define a line that passes 5 units away from the origin
    insertion_point = np.array([0.0, 0.0, 0.0])
    trajectory_direction = np.array([0.0, 1.0, 0.0])  # Pointing along y-axis
    pedicle_center = np.array([5.0, 10.0, 0.0])  # 5 units to the right, 10 units forward
    
    # Expected distance: 5.0 (x-component of pedicle_center)
    expected_distance = 5.0
    
    # Calculate distance using our function
    try:
        from .CostFunctions import distance_cost, point_to_line_distance
        
        # Test distance_cost function
        distance = distance_cost(insertion_point, trajectory_direction, pedicle_center)
        logger.info(f"distance_cost result: {distance}")
        logger.info(f"Expected result: {expected_distance}")
        logger.info(f"Difference: {abs(distance - expected_distance)}")
        
        # Test point_to_line_distance function as well
        distance2 = point_to_line_distance(pedicle_center, insertion_point, trajectory_direction)
        logger.info(f"point_to_line_distance result: {distance2}")
        logger.info(f"Expected result: {expected_distance}")
        logger.info(f"Difference: {abs(distance2 - expected_distance)}")
        
    except Exception as e:
        logger.error(f"Error during cost function test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
    # Visualize the test case
    try:
        # Create a line source for the trajectory
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(insertion_point)
        end_point = insertion_point + trajectory_direction * 20.0  # 20 units long
        line_source.SetPoint2(end_point)
        line_source.Update()
        
        # Create model for trajectory
        trajectory_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TestTrajectory")
        trajectory_model.SetAndObservePolyData(line_source.GetOutput())
        
        # Create display node for trajectory
        display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        display_node.SetColor(0.0, 1.0, 0.0)  # Green
        display_node.SetLineWidth(2.0)
        trajectory_model.SetAndObserveDisplayNodeID(display_node.GetID())
        
        # Create a sphere source for pedicle center
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(pedicle_center)
        sphere_source.SetRadius(1.0)
        sphere_source.Update()
        
        # Create model for pedicle center
        center_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TestPedicleCenter")
        center_model.SetAndObservePolyData(sphere_source.GetOutput())
        
        # Create display node for pedicle center
        center_display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        center_display.SetColor(1.0, 0.0, 0.0)  # Red
        center_model.SetAndObserveDisplayNodeID(center_display.GetID())
        
        # Calculate closest point on trajectory to pedicle center
        normalized_direction = trajectory_direction / np.linalg.norm(trajectory_direction)
        vec_to_center = pedicle_center - insertion_point
        projection = np.dot(vec_to_center, normalized_direction) * normalized_direction
        closest_point = insertion_point + projection
        
        # Create line from pedicle center to closest point (perpendicular distance)
        distance_line = vtk.vtkLineSource()
        distance_line.SetPoint1(pedicle_center)
        distance_line.SetPoint2(closest_point)
        distance_line.Update()
        
        # Create model for distance line
        distance_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TestDistanceLine")
        distance_model.SetAndObservePolyData(distance_line.GetOutput())
        
        # Create display node for distance line
        distance_display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        distance_display.SetColor(1.0, 1.0, 0.0)  # Yellow
        distance_display.SetLineWidth(2.0)
        distance_model.SetAndObserveDisplayNodeID(distance_display.GetID())
        
        # Change to 3D view
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        
        # Create a text annotation with the test results
        test_info = (
            f"Distance Cost Function Test\n"
            f"----------------------------\n"
            f"Insertion point: {insertion_point}\n"
            f"Trajectory direction: {trajectory_direction}\n"
            f"Pedicle center: {pedicle_center}\n"
            f"Expected distance: {expected_distance}\n"
            f"Calculated distance: {distance}\n"
            f"Difference: {abs(distance - expected_distance)}\n"
        )
        
        # Create a text node with the test info
        text_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", "TestDistanceCostInfo")
        text_node.SetText(test_info)
        
        # Display the text in the Python console
        logger.info("\n" + test_info)
        
        return {
            "trajectory_model": trajectory_model,
            "center_model": center_model,
            "distance_model": distance_model,
            "text_node": text_node,
            "distance": distance,
            "expected_distance": expected_distance
        }
        
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")