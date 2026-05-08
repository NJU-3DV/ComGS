import numpy as np
import cv2

import torch

from miscs import srgb_to_linear, adjust_exposure, TonemapHDR

scaler = np.array([0.212671, 0.715160, 0.072169])

# TODO: check the correctness
def process_image(image_dicts, indoor=True):
    """
    Process a dictionary of images to generate High Dynamic Range (HDR) and Low Dynamic Range (LDR) outputs.

    This function takes a dictionary of images in linear space, adjusts their exposure based on
    exposure values (EV), computes luminance values, blends them to create an HDR image, and
    applies tone mapping to produce an LDR image.

    Parameters
    ----------
    image_dicts : dict
        A dictionary where keys are exposure values (EV) as floats or integers, and values are
        image arrays (numpy.ndarray) in linear space. Each image is expected to have shape
        (height, width, channels), typically representing RGB data.

    Returns
    -------
    tuple
        A tuple containing:
        - hdr : numpy.ndarray
            The computed High Dynamic Range (HDR) image as a floating-point array with shape
            (height, width, channels).
        - ldr_img : numpy.ndarray
            The tone-mapped Low Dynamic Range (LDR) image as an 8-bit unsigned integer array
            with shape (height, width, channels) and pixel values in the range [0, 255].

    Notes
    -----
    - The input images are assumed to be in linear space (not gamma-corrected).
    - The exposure adjustment is performed by scaling the image intensity based on the negative
      of the exposure value (-EV).
    - Luminance is computed using a scaler (assumed to be a predefined vector or matrix for
      converting RGB to luminance).
    - The HDR image is blended using luminance masks to prioritize contributions from different
      exposure levels, avoiding overexposure.
    - Tone mapping is applied using a predefined `TonemapHDR` function with specific parameters
      (gamma=2.4, percentile=99, max_mapping=0.9) to convert HDR to LDR.

    Raises
    ------
    KeyError
        If the dictionary `image_dicts` does not contain a key for EV=0 or if any expected EV
        key is missing during processing.
    ValueError
        If the input images in `image_dicts` have inconsistent shapes or invalid data (e.g.,
        non-numeric values or unexpected dimensions).
    TypeError
        If `image_dicts` is not a dictionary or if the image arrays are not of type
        `numpy.ndarray`.

    Examples
    --------
    >>> image_dict = {0: np.ones((100, 100, 3)), -2: np.ones((100, 100, 3)) * 0.25}
    >>> hdr_img, ldr_img = process_image(image_dict)
    >>> hdr_img.shape
    (100, 100, 3)
    """
    # all image in image_dicts are in linear space
    evs = sorted(image_dicts.keys())

    image0 = image_dicts[0]
    luminaces = {}
    for _, ev in enumerate(evs):
        image = image_dicts[ev]
        image = adjust_exposure(image, -ev)
        lumi = image @ scaler
        luminaces[ev] = lumi

    out_luminace = luminaces[min(evs)]  # the darkest image
    for i, ev in enumerate(evs):
        maxval = np.pow(2, -ev)
        p1 = np.clip((luminaces[ev] - 0.9 * maxval) / (0.1 * maxval), 0, 1)
        p2 = out_luminace > luminaces[ev]
        mask = (p1 * p2).astype(np.float32)
        out_luminace = luminaces[ev] * (1 - mask) + out_luminace * mask

    hdr = image0 * (out_luminace / (luminaces[0] + 1e-10))[:, :, np.newaxis]
    
    # Compute luminance
    luminance = hdr @ scaler
    
    # Set the threshold and boost parameters
    threshold = np.mean(luminance) + 5 * np.std(luminance)  # Dynamic threshold
    # Maximum boost multiplier
    max_boost = 200.0 if indoor else 600.0  
    
    # Ensure the boosted region does not exceed 0.1%
    top_percentile = np.percentile(luminance, 99.99)  # 99.99th percentile
    
    # Create a smooth boost mask
    boost_factor = np.clip((luminance - threshold) / threshold, 0, 1)  # Boost factor in [0, 1]
    boost_factor = boost_factor ** 0.5  # Use a power curve for a smoother transition
    boost_multiplier = 1 + (max_boost - 1) * boost_factor
    boost_multiplier[luminance < top_percentile] = 1.0  # Do not boost values below the percentile threshold
    
    # Apply the boost
    hdr = hdr * boost_multiplier[:,:,np.newaxis]

    # luminance = hdr @ scaler

    # # Maximum boost multiplier. We assume the primary outdoor light source is the sun, so a stronger boost is needed.
    # max_boost = 50.0 if indoor else 100.0  

    # from test_light import analyze_light_sources, visualize_light_sources

    # light_sources = analyze_light_sources(luminance)
    # enhance_map = visualize_light_sources(luminance.shape, light_sources)

    # import matplotlib.pyplot as plt
    # plt.imshow(enhance_map, cmap='viridis')
    # plt.title('Fitted Light Sources')
    # plt.colorbar()
    # plt.show()
    
    # boost_multiplier = 1 + max_boost * enhance_map
    
    # hdr = hdr * boost_multiplier[:, :, np.newaxis]

    return hdr

def cartesian_to_polar(xyz):
    theta = np.arccos(np.clip(xyz[2], -1.0, 1.0))
    phi = np.arctan2(xyz[1], xyz[0])
    return phi, theta
    # return np.stack((phi, theta), axis=1)

def polar_to_cartesian(phi_theta):
    x = np.sin(phi_theta[:, 1]) * np.cos(phi_theta[:, 0])
    y = np.sin(phi_theta[:, 1]) * np.sin(phi_theta[:, 0])
    z = np.cos(phi_theta[:, 1])
    return np.stack((x, y, z), axis=1)

def sphere_points(n=128):
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    # xyz = points
    # x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return points

def convert_to_panorama(dirs, sizes, colors, resolution=512):
    grid_latitude, grid_longitude = torch.meshgrid(
        [torch.arange(resolution, dtype=torch.float), torch.arange(2 * resolution, dtype=torch.float)])
    grid_latitude = grid_latitude.add(0.5)
    grid_longitude = grid_longitude.add(0.5)
    grid_latitude = grid_latitude.mul(np.pi / resolution)
    grid_longitude = grid_longitude.mul(np.pi / resolution)

    x = torch.sin(grid_latitude) * torch.cos(grid_longitude)
    y = torch.sin(grid_latitude) * torch.sin(grid_longitude)
    z = torch.cos(grid_latitude)
    xyz =  torch.stack((x, y, z)).cuda()

    nbatch = colors.shape[0]
    lights = torch.zeros((nbatch, 3, resolution, resolution * 2), dtype=dirs.dtype, device=dirs.device)
    _, tmp = colors.shape
    nlights = int(tmp / 3)
    for i in range(nlights):
        lights = lights + (colors[:, 3 * i + 0:3 * i + 3][:, :, None, None]) * (
            torch.exp(
                (torch.matmul(dirs[:, 3 * i + 0:3 * i + 3], xyz.view(3, -1)).
                 view(-1, xyz.shape[1], xyz.shape[2]) - 1) /
                (sizes[:, i]).view(-1, 1, 1))[:, None, :, :])
    return lights



class extractEMLight():
    def __init__(self, h=128, w=256, ln=64):
        self.h, self.w = h, w
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]

        y_ = np.linspace(0, np.pi, num=h)  # + np.pi / h
        x_ = np.linspace(0, 2 * np.pi, num=w)  # + np.pi * 2 / w
        X, Y = np.meshgrid(x_, y_)
        Y = Y.reshape((-1, 1))
        X = X.reshape((-1, 1))
        phi_theta = np.stack((X, Y), axis=1)
        xyz = polar_to_cartesian(phi_theta)
        xyz = xyz.reshape((h, w, 3))  # 128, 256, 3
        xyz = np.expand_dims(xyz, axis=2)
        self.xyz = np.repeat(xyz, ln, axis=2)
        self.anchors = sphere_points(ln)

        dis_mat = np.linalg.norm((self.xyz - self.anchors), axis=-1)
        self.idx = np.argsort(dis_mat, axis=-1)[:, :, 0]
        self.ln, _ = self.anchors.shape

    def compute(self, hdr):

        hdr = self.steradian * hdr

        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[..., 1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]

        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=-1)
        light = hdr * map
        remain = hdr * (1 - map)

        ambient = remain.sum(axis=(0, 1))    #mean(axis=0).mean(axis=0)
        anchors = np.zeros((self.ln, 3))

        for i in range(self.ln):
            mask = self.idx == i
            mask = np.expand_dims(mask, -1)
            anchors[i] = (light * mask).sum(axis=(0, 1))

        anchors_engergy = 0.3 * anchors[..., 0] + 0.59 * anchors[..., 1] + 0.11 * anchors[..., 2]
        distribution = anchors_engergy / anchors_engergy.sum()
        anchors_rgb = anchors.sum(0)   # energy
        intensity = np.linalg.norm(anchors_rgb)
        rgb_ratio = anchors_rgb / intensity
        # distribution = anchors / intensity

        parametric_lights = {"distribution": distribution,
                             'intensity': intensity,
                             'rgb_ratio': rgb_ratio,
                             'ambient': ambient}
        return parametric_lights, map


def process_image2(image_dicts, type="indoor"):
    
    # all image in image_dicts are in linear space
    evs = sorted(image_dicts.keys())

    image0 = image_dicts[0]
    luminaces = {}
    for _, ev in enumerate(evs):
        image = image_dicts[ev]
        image = adjust_exposure(image, -ev)

        lumi = image @ scaler
        luminaces[ev] = lumi

    out_luminace = luminaces[min(evs)]  # the darkest image
    for i, ev in enumerate(evs):
        maxval = np.power(2, -ev)
        p1 = np.clip((luminaces[ev] - 0.9 * maxval) / (0.1 * maxval), 0, 1)
        p2 = out_luminace > luminaces[ev]
        mask = (p1 * p2).astype(np.float32)
        out_luminace = luminaces[ev] * (1 - mask) + out_luminace * mask

    hdr = image0 * (out_luminace / (luminaces[0] + 1e-10))[:, :, np.newaxis]

    # enhance light sources
    luminance = hdr @ scaler
    threshold = np.mean(luminance) + 5 * np.std(luminance)
    condition = luminance > threshold

    light_map = hdr * condition[..., np.newaxis]

    ln = 128
    extractor = extractEMLight(h=512, w=1024, ln=ln)
    para, _ = extractor.compute(light_map)

    dirs = sphere_points(ln)
    dirs = torch.from_numpy(dirs)
    dirs = dirs.view(1, ln*3).cuda().float()
    
    size = torch.ones((1, ln)).cuda().float() * 0.0025
    intensity = torch.from_numpy(np.array(para['intensity'])).float().cuda()
    intensity = intensity.view(1, 1, 1).repeat(1, ln, 3).cuda()
    
    rgb_ratio = torch.from_numpy(np.array(para['rgb_ratio'])).float().cuda()
    rgb_ratio = rgb_ratio.view(1, 1, 3).repeat(1, ln, 1).cuda()
    
    distribution = torch.from_numpy(para['distribution']).cuda().float()
    distribution = distribution.view(1, ln, 1).repeat(1, 1, 3)
    
    light_rec = distribution * intensity * rgb_ratio
    light_rec = light_rec.contiguous().view(1, ln*3)
    
    lgt = convert_to_panorama(dirs, size, light_rec)
    lgt = lgt.detach().cpu().numpy()[0]
    lgt = np.transpose(lgt, (1, 2, 0))

    lgt_illumination = lgt @ scaler

    max_boost = 5.0
    top_percentile = np.percentile(lgt_illumination, 99.99)
    lgt[lgt_illumination >= top_percentile] *= max_boost
    lgt[lgt_illumination < top_percentile] = 0.0

    return hdr, lgt

def merge_to_hdr_cv2(image_dicts):
    """
    Merge multiple images with different exposures into a single HDR image.

    Parameters
    ----------
    image_dicts : dict
        A dictionary where keys are exposure values (EV) as floats or integers, and values are
        image arrays (numpy.ndarray) in linear space. Each image is expected to have shape
        (height, width, channels), typically representing RGB data.

    Returns
    -------
    numpy.ndarray
        The computed High Dynamic Range (HDR) image as a floating-point array with shape
        (height, width, channels).
    """
    exposure_values = sorted(image_dicts.keys())
    exposure_times = np.array([2.0**(-ev) for ev in exposure_values], dtype=np.float32)
    images = [(image_dicts[ev] * 255.0).astype(np.uint8) for ev in exposure_values]
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, exposure_times)
    merge = cv2.createMergeDebevec()
    hdr = merge.process(images, exposure_times, response)
    return hdr

def tent_weight_function(pixel_values_normalized: np.ndarray) -> np.ndarray:
    """
    Compute weights that favor mid-range values and suppress values near 0 and 1.
    pixel_values_normalized: NumPy array with values in the range [0, 1].
    """
    # W(z) = 1 - |2z - 1|
    return 1.0 - np.abs(2.0 * pixel_values_normalized - 1.0)

def merge_to_hdr(exposure_dict: dict) -> np.ndarray:
    """
    Merge multiple sRGB NumPy images with different exposures into a single HDR image.

    Args:
        exposure_dict (dict): A dictionary where:
            - key (float): The image exposure value (EV).
            - value (np.ndarray): The corresponding sRGB image data as a NumPy array
                                  of type float32, with pixel values normalized to [0, 1]
                                  and shape (height, width, channels).

    Returns:
        numpy.ndarray: The merged HDR image (float32, linear space).
    """
    if not exposure_dict:
        raise ValueError("The exposure_dict must not be empty.")

    ev_values = sorted(exposure_dict.keys())
    srgb_images_normalized = [exposure_dict[ev] for ev in ev_values]

    if not srgb_images_normalized:
        raise ValueError("The exposure_dict does not contain any image data.")

    # --- Core logic begins ---
    exposure_times = np.array([2.0**(-ev) for ev in ev_values], dtype=np.float32)
    

    # Get the shape from the first image
    # Assume all images in the dictionary share the same shape
    first_image = srgb_images_normalized[0]
    img_shape = first_image.shape
    
    hdr_numerator = np.zeros(img_shape, dtype=np.float32)
    hdr_denominator = np.zeros(img_shape, dtype=np.float32)
    
    epsilon = 1e-9 # A small constant to avoid division by zero

    for i in range(len(srgb_images_normalized)):
        img_srgb_normalized = srgb_images_normalized[i] # This is already a float32 NumPy array in the range [0, 1]
        delta_t_i = exposure_times[i]
        

        weights = tent_weight_function(img_srgb_normalized)
        
        img_linear = srgb_to_linear(img_srgb_normalized)
        
        hdr_numerator += weights * (img_linear / delta_t_i)
        hdr_denominator += weights
        
    hdr_image = hdr_numerator / (hdr_denominator + epsilon)
    
    return hdr_image


if __name__ == "__main__":

    import imageio.v2 as imageio

    ev_images = r"D:\project\3dgsenvmap\3dgsenvmap\output\emlight\forest_with_kettle_000\inference\ev\batch0"
    
    image_dicts = {}
    for ev in [0, -2.5, -5]:
        image = imageio.imread(f"{ev_images}/{ev}.png")
        image = image.astype(np.float32) / 255.0
        image_dicts[ev] = image
    
    hdr = process_image2(image_dicts)

    from utils import save_exr
    save_exr("test.exr", hdr)
