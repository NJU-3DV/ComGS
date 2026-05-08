
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_composition_name(composition_name: str):
    """Parse the composition name into a scene and list of objects.
    
    Args:
        composition_name (str): Format like "scene_with_object1_and_object2"
    
    Returns:
        tuple: (scene, [object1, object2, ...])
    """
    parts = composition_name.split("_with_")
    if len(parts) != 2:
        print(f"Warning: composition name '{composition_name}' doesn't follow the expected format 'scene_with_objects'")
        return composition_name, []
    
    scene = parts[0]
    objects_part = parts[1]
    
    # Split objects by '_and_' or just '_and' if it's at the end
    objects = []
    remaining = objects_part
    
    while remaining:
        if '_and_' in remaining:
            split_idx = remaining.find('_and_')
            objects.append(remaining[:split_idx])
            remaining = remaining[split_idx + 5:]  # length of '_and_'
        elif remaining.endswith('_and'):
            objects.append(remaining[:-4])  # remove '_and'
            break
        else:
            objects.append(remaining)
            break
    
    return scene, objects

def get_object_transform(obj, transform_data):
    """
    Get the 4x4 transform matrix for an object from transform data.
    
    Args:
        obj (str): The object name
        transform_data (dict): Dictionary containing transform information
        
    Returns:
        numpy.ndarray: 4x4 transform matrix
    """
    if obj in transform_data:

        obj_transform = transform_data[obj]

        if obj_transform.get("transform", None) is not None:
            transform = np.array(obj_transform.get("transform"), dtype=np.float32).reshape(4, 4)
        elif obj_transform.get("transformation", None) is not None:
            transform = np.array(obj_transform.get("transformation"), dtype=np.float32).reshape(4, 4)
        else:
            # Extract transform components
            # Get transform components with more detailed logging
            rotation = obj_transform.get('rotation', [0.0, 0.0, 0.0])
            location = obj_transform.get('location', [0.0, 0.0, 0.0])
            scale = obj_transform.get('scale', [1.0, 1.0, 1.0])
            rotation_type = obj_transform.get('rotation_type', 'XYZ')
            
            rot = R.from_euler(rotation_type.lower(), rotation)
            rot_matrix = rot.as_matrix()
            
            # Build 4x4 transform matrix
            transform = np.eye(4)
            transform[:3, :3] = rot_matrix * np.array(scale)[:, None]
            transform[:3, 3] = location
    else:
        # Default transform if not found
        transform = np.eye(4)
    
    return transform