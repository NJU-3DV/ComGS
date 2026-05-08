
import os
import argparse
import numpy as np
from utils.read_write_model import read_model, write_model, Image, Point3D
from utils.read_write_model import rotmat2qvec, qvec2rotmat


def euler_to_rotation_matrix(euler_angles, sequence='XYZ'):
    """
    Converts Euler angles to a rotation matrix.

    Args:
        euler_angles (list or numpy.ndarray): Euler angles in degrees, in the order specified by 'sequence'.
        sequence (str): The order of the Euler angles ('XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX').  Default is 'XYZ'.

    Returns:
        numpy.ndarray: The 3x3 rotation matrix.
    """
    # Convert to radians
    angles = np.radians(euler_angles)

    # Create rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])

    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])

    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])

    # Combine rotations based on the sequence
    if sequence == 'XYZ':
        R = Rz @ Ry @ Rx
    elif sequence == 'XZY':
        R = Ry @ Rz @ Rx
    elif sequence == 'YXZ':
        R = Rz @ Rx @ Ry
    elif sequence == 'YZX':
        R = Rx @ Rz @ Ry
    elif sequence == 'ZXY':
        R = Ry @ Rx @ Rz
    elif sequence == 'ZYX':
        R = Rx @ Ry @ Rz
    else:
        raise ValueError("Invalid sequence. Choose from 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', or 'ZYX'.")

    return R

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", "-s", required=True, type=str,
                        help="Path to the COLMAP model.")
    parser.add_argument("--type", "-t", type=str, default="XYZ",
                        choices=["XYZ", "XZY", "YXZ", "YZX", "ZYX", "ZXY"],
                        help="Axis order for Euler angle rotations.")
    # 3. Rotation Angles (three floats for rotation around X, Y, Z axes)
    parser.add_argument("--angles", "-a", nargs=3, type=float, default=[-90.0, 0.0, 0.0],
                        help="Rotation angles (in degrees) around X, Y, Z axes, e.g., --angles 10 20 30")
    args = parser.parse_args()
    
    input_model_dir = os.path.join(args.source_path, "sparse/align")
    sparse_model_dir = os.path.join(args.source_path, "sparse/0")

    # read model
    cameras, images, points3D = read_model(input_model_dir)

    # deal with rotation...
    Rmat = euler_to_rotation_matrix(args.angles, args.type)
    RInv = Rmat.T

    # print(Rmat, RInv)
    # exit(-1)

    aligned_images = dict() 
    for idx, image in images.items():
        qvec = image.qvec

        r = qvec2rotmat(qvec)
        r_align = r @ RInv
        qvec_align = rotmat2qvec(r_align)
        
        aligned_images[idx] = Image(
            id = idx,
            qvec = qvec_align,
            tvec = image.tvec,
            camera_id = image.camera_id,
            name = image.name,
            xys = image.xys,
            point3D_ids = image.point3D_ids
        )
    
    aligned_point3D = dict()
    for idx, pt in points3D.items():
        xyz = pt.xyz

        xyz_align = Rmat @ xyz

        aligned_point3D[idx] = Point3D(
            id = pt.id,
            xyz = xyz_align,
            rgb = pt.rgb,
            error = pt.error,
            image_ids = pt.image_ids,
            point2D_idxs = pt.point2D_idxs
        )

    # save model
    os.makedirs(sparse_model_dir, exist_ok=True)
    write_model(cameras, aligned_images, aligned_point3D, sparse_model_dir)
