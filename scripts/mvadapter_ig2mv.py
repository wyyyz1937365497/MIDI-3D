# copied from https://github.com/huanngzh/MV-Adapter/blob/main/scripts/inference_ig2mv_partial_sdxl.py  
import argparse
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline  # 使用正确的SD pipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import make_image_grid, tensor_to_image
from mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper,
    get_orthogonal_camera,
    load_mesh,
    render,
)
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

# 添加全局异常处理以兼容旧版模型文件
try:
    # 尝试添加安全全局变量以支持某些模型文件
    import pytorch_lightning.callbacks.model_checkpoint
    torch.serialization.add_safe_globals([pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint])
except ImportError:
    pass

# 为旧版PyTorch兼容性准备weights_only参数
LOAD_KWARGS = {}


def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    lora_model,
    adapter_path,
    scheduler,
    num_views,
    device,
    dtype,
):
    # Load vae and unet if provided
    pipe_kwargs = {}
    vae_loaded = False
    unet_loaded = False
    
    if vae_model is not None:
        # 检查是否为本地路径（文件存在且有适当的扩展名）
        if os.path.exists(vae_model) and (vae_model.endswith('.pt') or vae_model.endswith('.ckpt') or vae_model.endswith('.safetensors')):
            print(f"Loading VAE from local path: {vae_model}")
            try:
                pipe_kwargs["vae"] = AutoencoderKL.from_single_file(vae_model, **LOAD_KWARGS)
                vae_loaded = True
            except Exception as e:
                print(f"Failed to load VAE with from_single_file: {e}")
                print("Continuing without custom VAE")
        else:
            print(f"Loading VAE from HF repo: {vae_model}")
            try:
                pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
                vae_loaded = True
            except Exception as e:
                print(f"Failed to load VAE from HF repo: {e}")
                print("Continuing without custom VAE")
            
    if unet_model is not None:
        # 检查是否为本地路径（文件存在且有适当的扩展名）
        if os.path.exists(unet_model) and (unet_model.endswith('.pt') or unet_model.endswith('.ckpt') or unet_model.endswith('.safetensors')):
            print(f"Loading UNet from local path: {unet_model}")
            try:
                pipe_kwargs["unet"] = UNet2DConditionModel.from_single_file(unet_model, **LOAD_KWARGS)
                unet_loaded = True
            except Exception as e:
                print(f"Failed to load UNet with from_single_file: {e}")
                print("Continuing without custom UNet")
        else:
            print(f"Loading UNet from HF repo: {unet_model}")
            try:
                pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)
                unet_loaded = True
            except Exception as e:
                print(f"Failed to load UNet from HF repo: {e}")
                print("Continuing without custom UNet")

    # Prepare pipeline - 使用SD2.1的pipeline
    pipe: MVAdapterI2MVSDPipeline
    # 检查基础模型是否为本地路径
    if os.path.exists(base_model) and (base_model.endswith(".ckpt") or base_model.endswith(".safetensors")):
        print(f"Loading base model from local checkpoint: {base_model}")
        try:
            # 当使用自定义组件时，我们只传递这些组件，避免加载基础模型中的对应组件
            pipe = MVAdapterI2MVSDPipeline.from_single_file(base_model, **pipe_kwargs, **LOAD_KWARGS)
        except Exception as e:
            print(f"Failed to load base model with from_single_file: {e}")
            # 如果已经加载了自定义组件，则创建一个不带这些组件的基础管道，然后手动设置组件
            if vae_loaded or unet_loaded:
                print("Custom components loaded, setting them manually...")
                # 先创建不带自定义组件的管道
                temp_kwargs = {}
                try:
                    pipe = MVAdapterI2MVSDPipeline.from_single_file(base_model, **temp_kwargs, **LOAD_KWARGS)
                except Exception as e_temp:
                    print(f"Failed to load base model with from_single_file (without custom components): {e_temp}")
                    # 如果仍然失败，尝试使用 from_pretrained
                    try:
                        pipe = MVAdapterI2MVSDPipeline.from_pretrained(base_model, **temp_kwargs)
                    except Exception as e_pretrained:
                        print(f"Failed to load base model with from_pretrained: {e_pretrained}")
                        raise e  # 如果所有方法都失败，则抛出原始异常
                
                # 然后手动设置自定义组件
                if vae_loaded:
                    pipe.vae = pipe_kwargs["vae"]
                if unet_loaded:
                    pipe.unet = pipe_kwargs["unet"]
            else:
                print("Trying from_pretrained without custom components")
                pipe_kwargs = {}  # 清空自定义组件
                try:
                    pipe = MVAdapterI2MVSDPipeline.from_pretrained(base_model, **pipe_kwargs)
                except Exception as e2:
                    print(f"Failed to load base model with from_pretrained: {e2}")
                    raise e  # 如果两种方法都失败，则抛出原始异常
    else:
        print(f"Loading base model from HF repo: {base_model}")
        try:
            pipe = MVAdapterI2MVSDPipeline.from_pretrained(base_model, **pipe_kwargs)
        except Exception as e:
            print(f"Failed to load base model from HF repo: {e}")
            # 如果已经加载了自定义组件，则创建一个不带这些组件的基础管道，然后手动设置组件
            if vae_loaded or unet_loaded:
                print("Custom components loaded, setting them manually...")
                # 先创建不带自定义组件的管道
                temp_kwargs = {}
                try:
                    pipe = MVAdapterI2MVSDPipeline.from_pretrained(base_model, **temp_kwargs)
                except Exception as e_pretrained:
                    print(f"Failed to load base model with from_pretrained (without custom components): {e_pretrained}")
                    raise e  # 如果仍然失败，则抛出原始异常
                
                # 然后手动设置自定义组件
                if vae_loaded:
                    pipe.vae = pipe_kwargs["vae"]
                if unet_loaded:
                    pipe.unet = pipe_kwargs["unet"]
            else:
                # 尝试不带自定义组件加载
                pipe_kwargs = {}
                try:
                    pipe = MVAdapterI2MVSDPipeline.from_pretrained(base_model, **pipe_kwargs)
                except Exception as e_no_kwargs:
                    print(f"Failed to load base model with from_pretrained (no kwargs): {e_no_kwargs}")
                    raise e  # 如果仍然失败，则抛出原始异常

    # Load scheduler if provided
    scheduler_class = None
    if scheduler == "ddpm":
        scheduler_class = DDPMScheduler
    elif scheduler == "lcm":
        scheduler_class = LCMScheduler
    # 注意：其他调度器类型需要在运行时设置，因为它们需要访问pipeline.scheduler

    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=scheduler_class,
    )
    pipe.init_custom_adapter(
        num_views=num_views, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
    )
    # 加载SD版本的adapter权重
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_ig2mv_sd21.safetensors"  # 使用SD版本的权重文件
    )

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    # load lora if provided
    if lora_model is not None:
        # 检查是否指定了具体的文件名
        if "/" in lora_model and "." in lora_model.split("/")[-1]:
            # 包含文件扩展名，按原来的方式处理
            model_, name_ = lora_model.rsplit("/", 1)
            try:
                pipe.load_lora_weights(model_, weight_name=name_, **LOAD_KWARGS)
            except Exception as e:
                print(f"Failed to load LoRA weights: {e}")
        else:
            # 不包含具体文件名，直接加载整个repo
            try:
                pipe.load_lora_weights(lora_model, **LOAD_KWARGS)
            except Exception as e:
                print(f"Failed to load LoRA weights: {e}")

    pipe.enable_vae_slicing()

    return pipe


def remove_bg(image, net, transform, device):
    image_size = image.size
    input_images = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = net(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image


def preprocess_image(image: Image.Image, height, width):
    image = np.array(image)
    alpha = image[..., 3] > 0
    H, W = alpha.shape
    # get the bounding box of alpha
    y, x = np.where(alpha)
    y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
    x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
    image_center = image[y0:y1, x0:x1]
    # resize the longer side to H * 0.9
    H, W, _ = image_center.shape
    if H > W:
        W = int(W * (height * 0.9) / H)
        H = int(height * 0.9)
    else:
        H = int(H * (width * 0.9) / W)
        W = int(width * 0.9)
    image_center = np.array(Image.fromarray(image_center).resize((W, H)))
    # pad to H, W
    start_h = (height - H) // 2
    start_w = (width - W) // 2
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[start_h : start_h + H, start_w : start_w + W] = image_center
    image = image.astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def run_pipeline(
    pipe,
    mesh_path,
    num_views,
    text,
    image,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed,
    remove_bg_fn=None,
    reference_conditioning_scale=1.0,
    negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
    lora_scale=1.0,
    device="cuda",
    scheduler=None,
):
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        device=device,
    )
    ctx = NVDiffRastContextWrapper(device=device)

    mesh, offset, scale = load_mesh(
        mesh_path,
        rescale=True,
        move_to_center=True,
        device=device,
        return_transform=True,
    )
    transform_dict = {"offset": offset.tolist(), "scale": scale.tolist()}

    render_out = render(
        ctx,
        mesh,
        cameras,
        height=height,
        width=width,
        render_attr=False,
        normal_background=0.0,
    )
    pos_images = tensor_to_image((render_out.pos + 0.5).clamp(0, 1), batched=True)
    normal_images = tensor_to_image(
        (render_out.normal / 2 + 0.5).clamp(0, 1), batched=True
    )
    control_images = (
        torch.cat(
            [
                (render_out.pos + 0.5).clamp(0, 1),
                (render_out.normal / 2 + 0.5).clamp(0, 1),
            ],
            dim=-1,
        )
        .permute(0, 3, 1, 2)
        .to(device)
    )

    # Prepare image
    reference_image = Image.open(image) if isinstance(image, str) else image
    if remove_bg_fn is not None:
        reference_image = remove_bg_fn(reference_image)
        reference_image = preprocess_image(reference_image, height, width)
    elif reference_image.mode == "RGBA":
        reference_image = preprocess_image(reference_image, height, width)

    # 设置采样器
    if scheduler is not None and scheduler != "Default":
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler
        )
        
        if scheduler == "Euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "Euler a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "DPM++":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "DPM++ SDE":
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=reference_image,
        reference_conditioning_scale=reference_conditioning_scale,
        negative_prompt=negative_prompt,
        cross_attention_kwargs={"scale": lora_scale},
        **pipe_kwargs,
        **LOAD_KWARGS,
    ).images

    return images, pos_images, normal_images, reference_image, transform_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Models - 修改为SD2.1的基础模型
    parser.add_argument(
        "--base_model", type=str, default="/home/wyyyz/.cache/modelscope/hub/models/stabilityai/stable-diffusion-2-1-base"
    )
    parser.add_argument(
        "--vae_model", type=str, default=None  # SD2.1使用内置VAE，不需要额外指定
    )
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--num_views", type=int, default=6)
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Inference
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, required=False, default="high quality")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)  # SD2.1通常使用7.5
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--reference_conditioning_scale", type=float, default=1.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="watermark, ugly, deformed, noisy, blurry, low contrast",
    )
    parser.add_argument("--output", type=str, default="output.png")
    # Extra
    parser.add_argument("--remove_bg", action="store_true", help="Remove background")
    args = parser.parse_args()

    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        lora_model=args.lora_model,
        adapter_path=args.adapter_path,
        scheduler=args.scheduler,
        num_views=args.num_views,
        device=args.device,
        dtype=torch.float16,
    )

    if args.remove_bg:
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        birefnet.to(args.device)
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, args.device)
    else:
        remove_bg_fn = None

    # 修改为SD2.1的分辨率 512x512
    images, pos_images, normal_images, reference_image, transform_dict = run_pipeline(
        pipe,
        mesh_path=args.mesh,
        num_views=args.num_views,
        text=args.text,
        image=args.image,
        height=512,  # SD2.1使用512x512分辨率
        width=512,   # SD2.1使用512x512分辨率
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        lora_scale=args.lora_scale,
        reference_conditioning_scale=args.reference_conditioning_scale,
        negative_prompt=args.negative_prompt,
        device=args.device,
        remove_bg_fn=remove_bg_fn,
    )
    make_image_grid(images, rows=1).save(args.output)
    make_image_grid(pos_images, rows=1).save(args.output.rsplit(".", 1)[0] + "_pos.png")
    make_image_grid(normal_images, rows=1).save(
        args.output.rsplit(".", 1)[0] + "_nor.png"
    )
    reference_image.save(args.output.rsplit(".", 1)[0] + "_reference.png")

    with open(args.output.rsplit(".", 1)[0] + "_transform.json", "w") as f:
        json.dump(transform_dict, f, indent=4)