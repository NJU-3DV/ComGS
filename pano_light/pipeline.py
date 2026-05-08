from diffusers import StableDiffusionPipeline, T2IAdapter
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.utils import deprecate
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
import torch.nn.functional as F 

class CustomPipeline(StableDiffusionPipeline):
    @torch.no_grad
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        image_input: Optional[PipelineImageInput] = None,
        image_mask: Optional[PipelineImageInput] = None,
        use_noise='just_latent',
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        eval_mode: bool = False,
        use_vae: bool = False,
        mode='nearest',
        **kwargs,
    ):
        r"""
            The call function to the pipeline for generation.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
                height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The height in pixels of the generated image.
                width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The width in pixels of the generated image.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                timesteps (`List[int]`, *optional*):
                    Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                    in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                    passed will be used. Must be in descending order.
                sigmas (`List[float]`, *optional*):
                    Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                    their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                    will be used.
                guidance_scale (`float`, *optional*, defaults to 7.5):
                    A higher guidance scale value encourages the model to generate images closely linked to the text
                    `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                    pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
                num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                eta (`float`, *optional*, defaults to 0.0):
                    Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                    to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
                generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                    A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                    generation deterministic.
                latents (`torch.Tensor`, *optional*):
                    Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor is generated by sampling using the supplied random `generator`.
                prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                    provided, text embeddings are generated from the `prompt` input argument.
                negative_prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                    not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
                ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                    Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                    IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                    contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                    provided, embeddings are computed from the `ip_adapter_image` input argument.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generated image. Choose between `PIL.Image` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                    plain tuple.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                    [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                guidance_rescale (`float`, *optional*, defaults to 0.0):
                    Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                    Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                    using zero terminal SNR.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
                callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                    A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                    each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                    DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                    list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
                callback_on_step_end_tensor_inputs (`List`, *optional*):
                    The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                    will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                    `._callback_tensor_inputs` attribute of your pipeline class.

            Returns:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                    If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                    otherwise a `tuple` is returned where the first element is a list with the generated images and the
                    second element is a list of `bool`s indicating whether the corresponding generated image contains
                    "not-safe-for-work" (nsfw) content.
            """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs


        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            assert height is not None and width is not None
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False


        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device


        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        ) # type: ignore

        # 5. Prepare latent variables
        if isinstance(image_input,torch.Tensor):
            image_input = image_input.to(device)
            image_input = image_input*2-1
            vae_image = self.vae.encode(image_input).latent_dist.sample()
            vae_image = vae_image*self.vae.config.scaling_factor

        if isinstance(image_mask,torch.Tensor):
            image_mask=image_mask.to(device) # [0, 1]
            image_mask_clone=image_mask.clone()
            if use_vae:
                image_mask = image_mask * 2 - 1 # [-1, 1]
                image_mask = self.vae.encode(image_mask).latent_dist.sample()
                image_mask = image_mask * self.vae.config.scaling_factor
            else:
                image_mask_in = F.interpolate(image_mask,(vae_image.shape[2],vae_image.shape[3]),mode=mode)
        else:
            raise ValueError(f"image mask must is torch.Tensor, But we get {type(image_mask)}")
        assert image_mask.max() > 0.5 and image_mask_in.max() > 0.5
        assert isinstance(image_mask_in, torch.Tensor)

        if use_noise=='with_noise':
            noise_latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                4,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            latents = torch.cat([noise_latents, image_mask_in, vae_image],dim=1) 
            if self.do_classifier_free_guidance:
                latents = torch.cat([latents] * 2) # 4 + 1 + 4 + 4 + 1 + 4
        else:
            raise ValueError(f"Unsupported use_noise value: {use_noise}. Only 'just_latent' is supported.")

        if eval_mode:
            latents = latents.detach()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if i>0:
                    if use_noise=='with_noise':
                        latents=torch.cat([noise_latents,image_mask_in,vae_image],dim=1)
                        if self.do_classifier_free_guidance:
                            latents = torch.cat([latents] * 2)
                        if eval_mode:
                            latents = latents.detach()
                            torch.cuda.empty_cache()


                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                torch.cuda.empty_cache()
                # predict the noise residual
                if kwargs.get("adapter", None) is not None:
                    t2iadapter = kwargs.get("adapter")
                    assert isinstance(t2iadapter, T2IAdapter)
                    weight_dtype = latent_model_input.dtype
                    conditioning_pixel_values = torch.cat([image_mask_clone]*3, dim=1) 
                    assert conditioning_pixel_values.shape[1] == 3
                    t2iadapter_image = conditioning_pixel_values.to(dtype=weight_dtype)
                    down_block_additional_residuals = t2iadapter(t2iadapter_image)
                    down_block_additional_residuals = [
                        sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals
                    ]
                    cross_block_additional_residuals = [
                        sample for sample in down_block_additional_residuals
                    ]

                    mask_map = (image_mask_clone>0.5)
                    mask_map = mask_map.float()
                    cross_attention_kwargs = {
                        'mask_map':mask_map,
                        "downblock_residual":cross_block_additional_residuals,
                    }  
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        # cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                        down_intrablock_additional_residuals=down_block_additional_residuals,
                    )[0]
                else:
                    mask_map = (image_mask_clone>0.5)
                    mask_map = mask_map.float()
                    cross_attention_kwargs = {
                        'mask_map': mask_map,
                    }
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        # cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if self.scheduler.config.prediction_type=='sample':
                    latents=noise_pred
                else: # self.scheduler.config.prediction_type=='v_prediction'
                    noise_latents = self.scheduler.step(noise_pred, t, noise_latents, **extra_step_kwargs, return_dict=False)[0]


                assert callback_on_step_end is None

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)


        if not output_type == "latent":
            if use_noise=='with_noise':
                latents=noise_latents.detach()
                # Ensure compatible shapes for broadcasting
            image=[]
            for index  in range(batch_size):

                torch.cuda.empty_cache()
                image_i = self.vae.decode(
                    latents[index:index+1] / self.vae.config.scaling_factor, 
                    return_dict=False,
                    generator=generator
                )[0]
                image_i, has_nsfw_concept = self.run_safety_checker(image_i, device, prompt_embeds.dtype)
                image.append(image_i)
            image=torch.cat(image,dim=0)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = image.detach()
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

class CustomPipelineNormal(StableDiffusionPipeline):
    @torch.no_grad
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        image_input: Optional[PipelineImageInput] = None,
        image_mask: Optional[PipelineImageInput] = None,
        image_normal: Optional[PipelineImageInput] = None,
        use_noise='just_latent',
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        eval_mode: bool = False,
        use_vae: bool = False,
        mode='nearest',
        **kwargs,
    ):
        r"""
            The call function to the pipeline for generation.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
                height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The height in pixels of the generated image.
                width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The width in pixels of the generated image.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                timesteps (`List[int]`, *optional*):
                    Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                    in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                    passed will be used. Must be in descending order.
                sigmas (`List[float]`, *optional*):
                    Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                    their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                    will be used.
                guidance_scale (`float`, *optional*, defaults to 7.5):
                    A higher guidance scale value encourages the model to generate images closely linked to the text
                    `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                    pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
                num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                eta (`float`, *optional*, defaults to 0.0):
                    Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                    to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
                generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                    A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                    generation deterministic.
                latents (`torch.Tensor`, *optional*):
                    Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor is generated by sampling using the supplied random `generator`.
                prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                    provided, text embeddings are generated from the `prompt` input argument.
                negative_prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                    not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
                ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                    Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                    IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                    contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                    provided, embeddings are computed from the `ip_adapter_image` input argument.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generated image. Choose between `PIL.Image` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                    plain tuple.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                    [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                guidance_rescale (`float`, *optional*, defaults to 0.0):
                    Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                    Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                    using zero terminal SNR.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
                callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                    A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                    each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                    DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                    list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
                callback_on_step_end_tensor_inputs (`List`, *optional*):
                    The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                    will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                    `._callback_tensor_inputs` attribute of your pipeline class.

            Returns:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                    If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                    otherwise a `tuple` is returned where the first element is a list with the generated images and the
                    second element is a list of `bool`s indicating whether the corresponding generated image contains
                    "not-safe-for-work" (nsfw) content.
            """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs


        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            assert height is not None and width is not None
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False


        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device


        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        ) # type: ignore

        # 5. Prepare latent variables
        if isinstance(image_input,torch.Tensor):
            image_input = image_input.to(device)
            image_input = image_input*2-1
            vae_image = self.vae.encode(image_input).latent_dist.sample()
            vae_image = vae_image*self.vae.config.scaling_factor

        if isinstance(image_mask,torch.Tensor):
            image_mask=image_mask.to(device) # [0, 1]
            image_mask_clone=image_mask.clone()
            if use_vae:
                image_mask = image_mask * 2 - 1 # [-1, 1]
                image_mask = self.vae.encode(image_mask).latent_dist.sample()
                image_mask = image_mask * self.vae.config.scaling_factor
            else:
                image_mask_in = F.interpolate(image_mask,(vae_image.shape[2],vae_image.shape[3]),mode=mode)
        else:
            raise ValueError(f"image mask must is torch.Tensor, But we get {type(image_mask)}")

        if isinstance(image_normal,torch.Tensor):
            image_normal=image_normal.to(device)
            image_normal_clone=image_normal.clone()
            image_normal_in = F.interpolate(image_normal,(vae_image.shape[2],vae_image.shape[3]),mode=mode)
        else:
            raise ValueError(f"image normal must is torch.Tensor, But we get {type(image_normal)}")

        if use_noise=='with_noise':
            noise_latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                4,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            latents = torch.cat([
                noise_latents, 
                image_normal_in,
                image_mask_in, 
                vae_image
            ],dim=1) 
            if self.do_classifier_free_guidance:
                latents = torch.cat([latents] * 2) # 4 + 1 + 4 + 4 + 1 + 4
        else:
            raise ValueError(f"Unsupported use_noise value: {use_noise}. Only 'just_latent' is supported.")

        if eval_mode:
            latents = latents.detach()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if i>0:
                    if use_noise=='with_noise':
                        latents=torch.cat([
                            noise_latents,
                            image_normal_in,
                            image_mask_in,
                            vae_image,
                        ],dim=1)
                        if self.do_classifier_free_guidance:
                            latents = torch.cat([latents] * 2)
                        if eval_mode:
                            latents = latents.detach()
                            torch.cuda.empty_cache()


                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                torch.cuda.empty_cache()
                # predict the noise residual
                if kwargs.get("adapter", None) is not None:
                    t2iadapter = kwargs.get("adapter")
                    assert isinstance(t2iadapter, T2IAdapter)
                    weight_dtype = latent_model_input.dtype
                    conditioning_pixel_values = torch.cat([image_normal_clone, image_mask_clone], dim=1)
                    assert conditioning_pixel_values.shape[1] == 4
                    t2iadapter_image = conditioning_pixel_values.to(dtype=weight_dtype)
                    down_block_additional_residuals = t2iadapter(t2iadapter_image)
                    down_block_additional_residuals = [
                        sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals
                    ]
                    cross_block_additional_residuals = [
                        sample for sample in down_block_additional_residuals
                    ]

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        # cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                        down_intrablock_additional_residuals=down_block_additional_residuals,
                    )[0]
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        # cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if self.scheduler.config.prediction_type=='sample':
                    latents=noise_pred
                else: # self.scheduler.config.prediction_type=='v_prediction'
                    noise_latents = self.scheduler.step(noise_pred, t, noise_latents, **extra_step_kwargs, return_dict=False)[0]


                assert callback_on_step_end is None

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)


        if not output_type == "latent":
            if use_noise=='with_noise':
                latents=noise_latents.detach()
                # Ensure compatible shapes for broadcasting
            image=[]
            for index  in range(batch_size):

                torch.cuda.empty_cache()
                image_i = self.vae.decode(
                    latents[index:index+1] / self.vae.config.scaling_factor, 
                    return_dict=False,
                    generator=generator
                )[0]
                image_i, has_nsfw_concept = self.run_safety_checker(image_i, device, prompt_embeds.dtype)
                image.append(image_i)
            image=torch.cat(image,dim=0)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = image.detach()
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


class CustomPipelineNormalAdapterNoMask(StableDiffusionPipeline):
    @torch.no_grad
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        image_input: Optional[PipelineImageInput] = None,
        image_mask: Optional[PipelineImageInput] = None,
        image_normal: Optional[PipelineImageInput] = None,
        use_noise='just_latent',
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        eval_mode: bool = False,
        use_vae: bool = False,
        mode='nearest',
        **kwargs,
    ):
        r"""
            The call function to the pipeline for generation.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
                height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The height in pixels of the generated image.
                width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The width in pixels of the generated image.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                timesteps (`List[int]`, *optional*):
                    Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                    in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                    passed will be used. Must be in descending order.
                sigmas (`List[float]`, *optional*):
                    Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                    their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                    will be used.
                guidance_scale (`float`, *optional*, defaults to 7.5):
                    A higher guidance scale value encourages the model to generate images closely linked to the text
                    `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                    pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
                num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                eta (`float`, *optional*, defaults to 0.0):
                    Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                    to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
                generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                    A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                    generation deterministic.
                latents (`torch.Tensor`, *optional*):
                    Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor is generated by sampling using the supplied random `generator`.
                prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                    provided, text embeddings are generated from the `prompt` input argument.
                negative_prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                    not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
                ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                    Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                    IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                    contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                    provided, embeddings are computed from the `ip_adapter_image` input argument.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generated image. Choose between `PIL.Image` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                    plain tuple.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                    [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                guidance_rescale (`float`, *optional*, defaults to 0.0):
                    Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                    Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                    using zero terminal SNR.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
                callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                    A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                    each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                    DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                    list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
                callback_on_step_end_tensor_inputs (`List`, *optional*):
                    The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                    will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                    `._callback_tensor_inputs` attribute of your pipeline class.

            Returns:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                    If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                    otherwise a `tuple` is returned where the first element is a list with the generated images and the
                    second element is a list of `bool`s indicating whether the corresponding generated image contains
                    "not-safe-for-work" (nsfw) content.
            """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs


        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            assert height is not None and width is not None
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False


        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device


        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        ) # type: ignore

        # 5. Prepare latent variables
        if isinstance(image_input,torch.Tensor):
            image_input = image_input.to(device)
            image_input = image_input*2-1
            vae_image = self.vae.encode(image_input).latent_dist.sample()
            vae_image = vae_image*self.vae.config.scaling_factor

        if isinstance(image_mask,torch.Tensor):
            image_mask=image_mask.to(device) # [0, 1]
            image_mask_clone=image_mask.clone()
            if use_vae:
                image_mask = image_mask * 2 - 1 # [-1, 1]
                image_mask = self.vae.encode(image_mask).latent_dist.sample()
                image_mask = image_mask * self.vae.config.scaling_factor
            else:
                image_mask_in = F.interpolate(image_mask,(vae_image.shape[2],vae_image.shape[3]),mode=mode)
        else:
            raise ValueError(f"image mask must is torch.Tensor, But we get {type(image_mask)}")

        if isinstance(image_normal,torch.Tensor):
            image_normal=image_normal.to(device)
            image_normal_clone=image_normal.clone()
            image_normal_in = F.interpolate(image_normal,(vae_image.shape[2],vae_image.shape[3]),mode=mode)
        else:
            raise ValueError(f"image normal must is torch.Tensor, But we get {type(image_normal)}")

        if use_noise=='with_noise':
            noise_latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                4,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            latents = torch.cat([
                noise_latents, 
                image_normal_in,
                image_mask_in, 
                vae_image
            ],dim=1) 
            if self.do_classifier_free_guidance:
                latents = torch.cat([latents] * 2) # 4 + 1 + 4 + 4 + 1 + 4
        else:
            raise ValueError(f"Unsupported use_noise value: {use_noise}. Only 'just_latent' is supported.")

        if eval_mode:
            latents = latents.detach()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if i>0:
                    if use_noise=='with_noise':
                        latents=torch.cat([
                            noise_latents,
                            image_normal_in,
                            image_mask_in,
                            vae_image,
                        ],dim=1)
                        if self.do_classifier_free_guidance:
                            latents = torch.cat([latents] * 2)
                        if eval_mode:
                            latents = latents.detach()
                            torch.cuda.empty_cache()


                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                torch.cuda.empty_cache()
                # predict the noise residual
                if kwargs.get("adapter", None) is not None:
                    t2iadapter = kwargs.get("adapter")
                    assert isinstance(t2iadapter, T2IAdapter)
                    weight_dtype = latent_model_input.dtype
                    conditioning_pixel_values = torch.cat([image_normal_clone], dim=1)
                    assert conditioning_pixel_values.shape[1] == 3
                    t2iadapter_image = conditioning_pixel_values.to(dtype=weight_dtype)
                    down_block_additional_residuals = t2iadapter(t2iadapter_image)
                    down_block_additional_residuals = [
                        sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals
                    ]

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                        down_intrablock_additional_residuals=down_block_additional_residuals,
                    )[0]
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if self.scheduler.config.prediction_type=='sample':
                    latents=noise_pred
                else: # self.scheduler.config.prediction_type=='v_prediction'
                    noise_latents = self.scheduler.step(noise_pred, t, noise_latents, **extra_step_kwargs, return_dict=False)[0]


                assert callback_on_step_end is None

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)


        if not output_type == "latent":
            if use_noise=='with_noise':
                latents=noise_latents.detach()
                # Ensure compatible shapes for broadcasting
            image=[]
            for index  in range(batch_size):

                torch.cuda.empty_cache()
                image_i = self.vae.decode(
                    latents[index:index+1] / self.vae.config.scaling_factor, 
                    return_dict=False,
                    generator=generator
                )[0]
                image_i, has_nsfw_concept = self.run_safety_checker(image_i, device, prompt_embeds.dtype)
                image.append(image_i)
            image=torch.cat(image,dim=0)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = image.detach()
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
