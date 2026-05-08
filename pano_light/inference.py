


import os

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import logging
import numpy as np
import cv2

import torch
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    T2IAdapter,
    UNet2DConditionModel,
    DDIMScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer

from pipeline import CustomPipelineNormalAdapterNoMask
from miscs import save_exr
from exposure_to_hdr import process_image2
from utils.seed_utils import make_numpy_rng, make_torch_generator, set_global_seed

import time

# prepare logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def resolve_local_model_path(path_or_repo: str) -> str:
    if os.path.isabs(path_or_repo):
        return path_or_repo

    candidate = os.path.join(current_dir, path_or_repo)
    if os.path.exists(candidate):
        return candidate
    return path_or_repo


# prepare get prompt_embeding function
def get_prompt_embeds(tokenizer, captions):
    token_output = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = token_output.input_ids
    return input_ids

def prepare_prompt_embeds(args, tokenizer, text_encoder, device):
    indoor_normal_prompt = args.indoor_normal_prompt
    indoor_dark_prompt = args.indoor_dark_prompt
    prompt_embeds_indoor_norm = text_encoder(
        get_prompt_embeds(tokenizer, indoor_normal_prompt).to(device), return_dict=False
    )[0]
    prompt_embeds_indoor_dark = text_encoder(
        get_prompt_embeds(tokenizer, indoor_dark_prompt).to(device), return_dict=False
    )[0]
    outdoor_normal_prompt = args.outdoor_normal_prompt
    outdoor_dark_prompt = args.outdoor_dark_prompt
    prompt_embeds_outdoor_norm = text_encoder(
        get_prompt_embeds(tokenizer, outdoor_normal_prompt).to(device), return_dict=False
    )[0]
    prompt_embeds_outdoor_dark = text_encoder(
        get_prompt_embeds(tokenizer, outdoor_dark_prompt).to(device), return_dict=False
    )[0]
    prompt_embeds_dict = {
        "prompt_embeds_indoor_norm": prompt_embeds_indoor_norm, 
        "prompt_embeds_indoor_dark": prompt_embeds_indoor_dark, 
        "prompt_embeds_outdoor_norm": prompt_embeds_outdoor_norm, 
        "prompt_embeds_outdoor_dark": prompt_embeds_outdoor_dark
    }
    return prompt_embeds_dict


def interpolate_embeding(args, ev, labels, prompt_embeds_dict):
    # ev and max_negative ev are all negative vales
    # max_negative_ev is the value of the abs of minium value
    # ev is the value of the abs of the truth ev
    interpolant = (ev / args.max_negative_ev).view(-1, 1, 1)  # [b, 1] -> [b, 1, 1]
    prompt_embeds_norm = torch.stack(
        [prompt_embeds_dict[f"prompt_embeds_{label}_norm"] for label in labels]
    ).squeeze()
    prompt_embeds_dark = torch.stack(
        [prompt_embeds_dict[f"prompt_embeds_{label}_dark"] for label in labels]
    ).squeeze()
    int_prompt_embeds = prompt_embeds_norm + interpolant * (prompt_embeds_dark - prompt_embeds_norm)
    return int_prompt_embeds

def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name, map_location='cpu')
    for prefix, model in models:
        assert isinstance(model, torch.nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])

    epoch = ckpt_dict['n_iter'] if 'n_iter' in ckpt_dict else 0
    step = ckpt_dict['step'] if 'step' in ckpt_dict else 0

    return step,  epoch

def get_saturated_regions(im, th=0.95):
    def saturated_channel_(im, th):
        return torch.clamp((im - th) / (1 - th), 0.0, 1.0)
    
    # Create output tensor with same shape as input
    mask_conv = torch.zeros_like(im)
    
    # Process each channel
    for i in range(im.shape[0]):
        mask_conv[i] = saturated_channel_(im[i], th)
        
    return mask_conv

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor([0.229, 0.224, 0.225]).to(x.device) + torch.Tensor([0.485, 0.456, 0.406]).to(x.device)
    x = x.transpose(1, 3)
    return x


@torch.no_grad()
def inference(image_tensor, mask_tensor, label, config, seed):
    """
    Inference function for HDR synthesis
    
    Args:
        image_tensor: torch.Tensor of shape [3, H, W], range [0, 1]
        mask_tensor: torch.Tensor of shape [H, W], range [0, 1]
        label: str, either "indoor" or "outdoor"
        config: static configuration arguments
        seed: runtime random seed
    
    Returns:
        dict: containing HDR results and intermediate outputs
    """
    set_global_seed(seed)

    # Load necessary models
    models_dict = load_models(config)
    
    # Get model components from loaded models
    device = models_dict['device']
    weight_dtype = models_dict['weight_dtype']
    pipeline = models_dict['pipeline']
    unet_dict = models_dict['unet_dict']
    t2i_adapter_dict = models_dict['t2i_adapter_dict']
    ev_dicts = models_dict['ev_dicts']
    
    # Convert input tensors to proper format
    image_tensor = image_tensor.to(device=device, dtype=weight_dtype)
    mask_tensor = mask_tensor.to(device=device, dtype=weight_dtype)
    
    # Ensure image_tensor is in [0, 1] range and has proper shape
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Convert image tensor to numpy for processing
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mask_np = mask_tensor.squeeze().cpu().numpy()
    
    # Process mask (same as original logic)
    if config.soft_mask:
        mask_processed = mask_np.astype(np.float32)
    else:
        mask_processed = (mask_np > 0.5).astype(np.float32)
    
    # Apply morphological operations to mask
    kernel = np.ones((7,7), np.uint8)
    mask_processed = cv2.erode(mask_processed, kernel, iterations=1)
    
    # Fill holes in the mask
    kernel = np.ones((5,5), np.uint8)
    mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel)
    mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, kernel)
    
    # Get the largest connected component
    # num_labels, _labels = cv2.connectedComponents(mask_processed.astype(np.uint8))
    # if num_labels > 1:
    #     sizes = [np.sum(_labels == i) for i in range(1, num_labels)]
    #     largest_label = np.argmax(sizes) + 1
    #     mask_processed = (_labels == largest_label).astype(np.float32)
    
    # Apply mask to image
    image_processed = image_np * mask_processed[..., np.newaxis]

    # Generate random normal map (as in original code)
    numpy_rng = make_numpy_rng(seed)
    normal_np = (2 * numpy_rng.random(image_np.shape) - 1).astype(image_np.dtype)
    normal_np = normal_np * mask_processed[..., np.newaxis]
    
    # Convert to tensors
    to_tensor = transforms.ToTensor()
    image_in = to_tensor(image_processed).unsqueeze(0).to(device=device, dtype=weight_dtype)
    image_mask = to_tensor(mask_processed).unsqueeze(0).to(device=device, dtype=weight_dtype)
    image_normal = to_tensor(normal_np).unsqueeze(0).to(device=device, dtype=weight_dtype)
    
    height, width = image_in.shape[2:]
    
    # Prepare generator
    generator = make_torch_generator(seed, device)
    
    # Prepare init noise
    shape = (
        1,
        4,
        int(height) // pipeline.vae_scale_factor,
        int(width) // pipeline.vae_scale_factor,
    )
    noise_latents = randn_tensor(shape, generator=generator, device=generator.device, dtype=weight_dtype)
    
    # Generate images for different exposure values
    start = time.time()
    exposure_dicts = {}
    for ev in config.evs:
        pipeline.unet = unet_dict[ev]
        encoder_hidden_states = ev_dicts[config.evs[0]][label]
        images_predicted = pipeline(
            prompt=None,
            prompt_embeds=encoder_hidden_states,
            num_inference_steps=50,
            image_input=image_in,
            image_mask=image_mask,
            image_normal=image_normal,
            generator=generator,
            latents=noise_latents,
            height=height,
            width=width,
            eta=0.1,
            use_noise="with_noise",
            adapter=t2i_adapter_dict[ev],
            return_dict=False,
            output_type="pt",
            use_vae=False,
            eval_mode=True,
            guidance_scale=7.5 if config.do_classifier_free_guidance else 1.0
        )[0]
        images_predicted_np = images_predicted.cpu().permute(0, 2, 3, 1).float().numpy()
        pred_img_srgb_np = images_predicted_np
        exposure_dicts[ev] = pred_img_srgb_np[0]
    
    # Convert exposure images to HDR
    hdr_synthetic, lgt_detected = process_image2(exposure_dicts, type=label)

    end = time.time()
    process_time = end - start
    
    return {
        'hdr_synthetic': hdr_synthetic,
        'lgt_detected': lgt_detected,
        'exposure_dicts': exposure_dicts,
        'image_in': image_in.cpu(),
        'mask_processed': mask_processed
    }, process_time


def load_models(args):
    """Load all models and return them in a dictionary"""
    # prepare device and weight_dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    pretrained_model_name_or_path = resolve_local_model_path(args.pretrained_model_name_or_path)
    pretrained_path_ev0 = resolve_local_model_path(args.pretrained_path_ev0)
    pretrained_path_ev25 = resolve_local_model_path(args.pretrained_path_ev25)
    pretrained_path_ev5 = resolve_local_model_path(args.pretrained_path_ev5)

    logger.info("Loading base diffusion checkpoint from: %s", pretrained_model_name_or_path)
    logger.info("Loading EV0 checkpoint from: %s", os.path.join(pretrained_path_ev0, "unet"))
    logger.info("Loading EV25 checkpoint from: %s", os.path.join(pretrained_path_ev25, "unet"))
    logger.info("Loading EV5 checkpoint from: %s", os.path.join(pretrained_path_ev5, "unet"))

    # load tokenizer and text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer", 
        revision=args.revision, 
        local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder", 
        revision=args.revision, 
        variant=args.variant,
        local_files_only=True,
    ).to(device, dtype=weight_dtype)

    # Load vae
    vae_path = (
        pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else resolve_local_model_path(args.pretrained_vae_model_name_or_path)
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device, dtype=weight_dtype)

    # Load unet
    unet_dict = {}
    unet_dict[0] = UNet2DConditionModel.from_pretrained(
        os.path.join(pretrained_path_ev0, "unet"),
        local_files_only=True,
    ).to(device, dtype=weight_dtype)
    unet_dict[-2.5] = UNet2DConditionModel.from_pretrained(
        os.path.join(pretrained_path_ev25, "unet"),
        local_files_only=True,
    ).to(device, dtype=weight_dtype)
    unet_dict[-5] = UNet2DConditionModel.from_pretrained(
        os.path.join(pretrained_path_ev5, "unet"),
        local_files_only=True,
    ).to(device, dtype=weight_dtype)

    # load t2iadapter
    t2i_adapter_dict = {}
    t2i_adapter_dict[0] = T2IAdapter.from_pretrained(
        os.path.join(pretrained_path_ev0, "t2iadapter"), 
        local_files_only=True,
    ).to(device, dtype=weight_dtype) 
    t2i_adapter_dict[-2.5] = T2IAdapter.from_pretrained(
        os.path.join(pretrained_path_ev25, "t2iadapter"), 
        local_files_only=True,
    ).to(device, dtype=weight_dtype) 
    t2i_adapter_dict[-5] = T2IAdapter.from_pretrained(
        os.path.join(pretrained_path_ev5, "t2iadapter"), 
        local_files_only=True,
    ).to(device, dtype=weight_dtype) 

    # Load scheduler
    scheduler_class = DDIMScheduler
    noise_scheduler = scheduler_class.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler", timestep_spacing="trailing",
        local_files_only=True,
    )
    noise_scheduler.config.prediction_type = 'sample'
    noise_scheduler_config = noise_scheduler.config
    noise_scheduler_config['beta_schedule'] = 'linear'
    noise_scheduler = scheduler_class.from_config(noise_scheduler_config)

    # load_pipepline
    pipeline = CustomPipelineNormalAdapterNoMask.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        unet=unet_dict[-5],
        scheduler=noise_scheduler,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipeline.scheduler.config.prediction_type='v_prediction'
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # prepare prompt embeding
    prompt_embeds_dict = prepare_prompt_embeds(
        args,
        tokenizer,
        text_encoder,
        device
    )
    ev_dicts = {}
    for ev in args.evs:
        ev_tensor = torch.tensor(ev).unsqueeze(0).to(device=device, dtype=weight_dtype)
        indoor_encoder_hidden_states = interpolate_embeding(args, ev_tensor, ["indoor"], prompt_embeds_dict)
        outdoor_encoder_hidden_states = interpolate_embeding(args, ev_tensor, ["outdoor"], prompt_embeds_dict)
        ev_dicts[ev] = {
            "indoor": indoor_encoder_hidden_states,
            "outdoor": outdoor_encoder_hidden_states
        }

    return {
        'device': device,
        'weight_dtype': weight_dtype,
        'pipeline': pipeline,
        'unet_dict': unet_dict,
        't2i_adapter_dict': t2i_adapter_dict,
        'ev_dicts': ev_dicts
    }
