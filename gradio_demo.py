import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()
# 后边正常写你的代码

import os
import random
import uuid
import gc
import psutil
import time
from typing import Any, List, Optional, Union, Dict
import torch.nn as nn

import gradio as gr
import numpy as np
import torch
import trimesh
from gradio_image_prompter import ImagePrompter
from huggingface_hub import snapshot_download
from PIL import Image

from midi.pipelines.pipeline_midi import MIDIPipeline
from scripts.grounding_sam import detect, plot_segmentation, prepare_model, segment
from scripts.image_to_textured_scene import (
    prepare_ig2mv_pipeline,
    prepare_texture_pipeline,
    run_i2tex,
)
from scripts.inference_midi import run_midi

# import spaces

# Constants
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "VAST-AI/MIDI-3D"
CHUNK_SIZE_MB = 500  # Size of each weight chunk in MB

MARKDOWN = """
## Image to 3D Scene with [MIDI-3D](https://huanngzh.github.io/MIDI-Page/)
1. Upload an image, and draw bounding boxes for each instance by holding and dragging the mouse, or use text labels to segment the image. Then click "Run Segmentation" to generate the segmentation result. <b>Nota that if you select "box" mode, ensure instances should not be too small and bounding boxes fit snugly around each instance.</b>
2. <b>Check "Do image padding" in "Generation Settings" if instances in your image are too close to the image border.</b> Then click "Run Generation" to generate a 3D scene from the image and segmentation result.
3. If you find the generated 3D scene satisfactory, download it by clicking the "Download GLB" button.
"""

EXAMPLES = [
    [
        {
            "image": "assets/example_data/Cartoon-Style/00_rgb.png",
        },
        "assets/example_data/Cartoon-Style/00_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Cartoon-Style/01_rgb.png",
        },
        "assets/example_data/Cartoon-Style/01_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Cartoon-Style/03_rgb.png",
        },
        "assets/example_data/Cartoon-Style/03_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/00_rgb.png",
        },
        "assets/example_data/Realistic-Style/00_seg.png",
        42,
        False,
        True,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/01_rgb.png",
        },
        "assets/example_data/Realistic-Style/01_seg.png",
        42,
        False,
        True,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/02_rgb.png",
        },
        "assets/example_data/Realistic-Style/02_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/05_rgb.png",
        },
        "assets/example_data/Realistic-Style/05_seg.png",
        42,
        False,
        False,
    ],
]

os.makedirs(TMP_DIR, exist_ok=True)

# Global variables to track model loading status
models_loaded = {
    "grounding_sam": False,
    "midi": False,
    "mv_adapter": False,
}

# Model containers - initially None
object_detector = None
sam_processor = None
sam_segmentator = None
pipe = None
ig2mv_pipe = None
texture_pipe = None

class ChunkedWeightLoader:
    """Manages chunked loading of model weights to minimize RAM usage"""
    
    def __init__(self, chunk_size_mb: int = CHUNK_SIZE_MB):
        self.chunk_size_mb = chunk_size_mb
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024
        
    def estimate_tensor_size(self, tensor: torch.Tensor) -> int:
        """Estimate the size of a tensor in bytes"""
        return tensor.numel() * tensor.element_size()
    
    def split_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Split state dict into chunks of specified size"""
        chunks = []
        current_chunk = {}
        current_size = 0
        
        for key, tensor in state_dict.items():
            tensor_size = self.estimate_tensor_size(tensor)
            
            # If adding this tensor would exceed chunk size, start a new chunk
            if current_size + tensor_size > self.chunk_size_bytes and current_chunk:
                chunks.append(current_chunk)
                current_chunk = {}
                current_size = 0
            
            current_chunk[key] = tensor
            current_size += tensor_size
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def load_model_chunked(self, model: nn.Module, state_dict_path: str, device: str = DEVICE, dtype: torch.dtype = DTYPE) -> None:
        """Load model weights in chunks to minimize RAM usage"""
        print(f"Loading model weights in chunks of {self.chunk_size_mb}MB...")
        
        # Load state dict on CPU first
        print("Loading state dict to CPU...")
        full_state_dict = torch.load(state_dict_path, map_location='cpu')
        
        # Split into chunks
        chunks = self.split_state_dict(full_state_dict)
        print(f"Split weights into {len(chunks)} chunks")
        
        # Clear the full state dict to free RAM
        del full_state_dict
        gc.collect()
        
        # Load chunks one by one
        for i, chunk in enumerate(chunks):
            print(f"Loading chunk {i+1}/{len(chunks)}...")
            
            # Load chunk to device with correct dtype
            chunk = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) 
                    for k, v in chunk.items()}
            
            # Load into model
            model.load_state_dict(chunk, strict=False)
            
            # Clear chunk from RAM
            del chunk
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Chunk {i+1} loaded successfully")
        
        print("All chunks loaded successfully")

def get_memory_info():
    """Get comprehensive memory usage information"""
    info = []
    
    # GPU Memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        info.append(f"GPU - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        info.append("GPU: Not Available")
    
    # System RAM
    process = psutil.Process(os.getpid())
    ram_info = process.memory_info()
    ram_used = ram_info.rss / 1024**3  # GB
    ram_percent = process.memory_percent()
    system_ram = psutil.virtual_memory()
    system_ram_used = system_ram.used / 1024**3
    system_ram_total = system_ram.total / 1024**3
    
    info.append(f"Process RAM: {ram_used:.2f}GB ({ram_percent:.1f}%)")
    info.append(f"System RAM: {system_ram_used:.1f}/{system_ram_total:.1f}GB ({system_ram.percent:.1f}%)")
    
    return "\n".join(info)

def aggressive_cleanup():
    """Aggressive memory cleanup for both GPU and RAM"""
    # Multiple rounds of garbage collection
    for _ in range(5):
        gc.collect()
        time.sleep(0.1)  # Give time for GC to work
    
    if torch.cuda.is_available():
        # Multiple rounds of CUDA cache clearing
        for _ in range(5):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(0.1)
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
    
    # Final garbage collection
    for _ in range(3):
        gc.collect()

def clear_model_attributes(model):
    """Recursively clear all attributes of a model to free RAM"""
    if model is None:
        return
    
    # Move to CPU first if it's on GPU
    if hasattr(model, 'to'):
        try:
            model.to('cpu')
        except:
            pass
    
    # Clear all possible attributes
    for attr in list(model.__dict__.keys()):
        try:
            setattr(model, attr, None)
        except:
            pass
    
    # If it's a nn.Module, clear its parameters and buffers
    if hasattr(model, 'parameters'):
        try:
            for param in model.parameters():
                param.data = None
        except:
            pass
    
    if hasattr(model, 'buffers'):
        try:
            for buffer in model.buffers():
                buffer.data = None
        except:
            pass

def load_grounding_sam():
    """Load Grounding SAM models on demand"""
    global object_detector, sam_processor, sam_segmentator, models_loaded
    
    if not models_loaded["grounding_sam"]:
        print("Loading Grounding SAM models...")
        print(f"Memory before loading:\n{get_memory_info()}")
        
        object_detector, sam_processor, sam_segmentator = prepare_model(
            device=DEVICE,
            detector_id="IDEA-Research/grounding-dino-tiny",
            segmenter_id="facebook/sam-vit-base",
        )
        models_loaded["grounding_sam"] = True
        
        print(f"Memory after loading:\n{get_memory_info()}")
        print("Grounding SAM models loaded successfully.")

def unload_grounding_sam():
    """Unload Grounding SAM models and aggressively free RAM"""
    global object_detector, sam_processor, sam_segmentator, models_loaded
    
    if models_loaded["grounding_sam"]:
        print("Unloading Grounding SAM models...")
        print(f"Memory before unloading:\n{get_memory_info()}")
        
        # Clear model attributes before deletion
        if object_detector is not None:
            clear_model_attributes(object_detector)
        if sam_segmentator is not None:
            clear_model_attributes(sam_segmentator)
        
        # Clear references
        object_detector = None
        sam_processor = None
        sam_segmentator = None
        models_loaded["grounding_sam"] = False
        
        # Aggressive cleanup
        aggressive_cleanup()
        
        print(f"Memory after unloading:\n{get_memory_info()}")
        print("Grounding SAM models unloaded and memory freed.")

def load_midi_model():
    """Load MIDI model on demand with chunked loading"""
    global pipe, models_loaded
    
    if not models_loaded["midi"]:
        print("Loading MIDI model with chunked weight loading...")
        print(f"Memory before loading:\n{get_memory_info()}")
        
        local_dir = "pretrained_weights/MIDI-3D"
        if not os.path.exists(local_dir):
            snapshot_download(repo_id=REPO_ID, local_dir=local_dir)
        
        # Initialize chunked loader
        chunked_loader = ChunkedWeightLoader(chunk_size_mb=CHUNK_SIZE_MB)
        
        # Create pipeline without loading weights
        print("Creating pipeline structure...")
        pipe = MIDIPipeline.from_pretrained(local_dir, torch_dtype=DTYPE)
        
        # Move pipeline to device
        pipe = pipe.to(DEVICE)
        
        # Ensure VAE is in float32 for stability, then convert to float16
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            print("Setting VAE to float32 for stability...")
            pipe.vae = pipe.vae.to(torch.float32)
        
        # Find the main model weights file
        weight_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file.endswith('.bin') or file.endswith('.pth') or file.endswith('.pt'):
                    weight_files.append(os.path.join(root, file))
        
        if weight_files:
            print(f"Found {len(weight_files)} weight files")
            
            # Load each weight file in chunks
            for weight_file in weight_files:
                print(f"Loading weights from {weight_file}...")
                try:
                    # Try to load chunked
                    chunked_loader.load_model_chunked(pipe, weight_file, DEVICE, DTYPE)
                except Exception as e:
                    print(f"Chunked loading failed for {weight_file}: {e}")
                    print("Falling back to normal loading...")
                    # Fallback to normal loading
                    state_dict = torch.load(weight_file, map_location=DEVICE)
                    # Convert to correct dtype
                    state_dict = {k: v.to(dtype=DTYPE) if v.is_floating_point() else v.to(DEVICE) 
                                 for k, v in state_dict.items()}
                    pipe.load_state_dict(state_dict, strict=False)
                    del state_dict
                    gc.collect()
        
        # Initialize custom adapter
        pipe.init_custom_adapter(
            set_self_attn_module_names=[
                "blocks.8",
                "blocks.9",
                "blocks.10",
                "blocks.11",
                "blocks.12",
            ]
        )
        
        # Convert the entire pipeline to float16 except VAE
        print("Converting pipeline to float16...")
        pipe = pipe.to(dtype=DTYPE)
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            # Keep VAE in float32 for numerical stability
            pipe.vae = pipe.vae.to(torch.float32)
        
        models_loaded["midi"] = True
        
        print(f"Memory after loading:\n{get_memory_info()}")
        print("MIDI model loaded successfully with chunked weights.")

def unload_midi_model():
    """Unload MIDI model and aggressively free RAM"""
    global pipe, models_loaded
    
    if models_loaded["midi"]:
        print("Unloading MIDI model...")
        print(f"Memory before unloading:\n{get_memory_info()}")
        
        if pipe is not None:
            # Move to CPU first
            try:
                pipe.to('cpu')
            except:
                pass
            
            # Clear all pipeline components
            components = ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler', 
                         'feature_extractor', 'image_encoder', 'safety_checker']
            
            for comp in components:
                if hasattr(pipe, comp) and getattr(pipe, comp) is not None:
                    clear_model_attributes(getattr(pipe, comp))
                    setattr(pipe, comp, None)
            
            # Clear the pipeline itself
            clear_model_attributes(pipe)
        
        # Clear reference
        pipe = None
        models_loaded["midi"] = False
        
        # Aggressive cleanup
        aggressive_cleanup()
        
        print(f"Memory after unloading:\n{get_memory_info()}")
        print("MIDI model unloaded and memory freed.")

def load_mv_adapter():
    """Load MV-Adapter models on demand"""
    global ig2mv_pipe, texture_pipe, models_loaded
    
    if not models_loaded["mv_adapter"]:
        print("Loading MV-Adapter models...")
        print(f"Memory before loading:\n{get_memory_info()}")
        
        ig2mv_pipe = prepare_ig2mv_pipeline(device="cuda", dtype=torch.float16)
        texture_pipe = prepare_texture_pipeline(device="cuda", dtype=torch.float16)
        models_loaded["mv_adapter"] = True
        
        print(f"Memory after loading:\n{get_memory_info()}")
        print("MV-Adapter models loaded successfully.")

def unload_mv_adapter():
    """Unload MV-Adapter models and aggressively free RAM"""
    global ig2mv_pipe, texture_pipe, models_loaded
    
    if models_loaded["mv_adapter"]:
        print("Unloading MV-Adapter models...")
        print(f"Memory before unloading:\n{get_memory_info()}")
        
        # Clear ig2mv_pipe
        if ig2mv_pipe is not None:
            try:
                ig2mv_pipe.to('cpu')
            except:
                pass
            
            components = ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
            for comp in components:
                if hasattr(ig2mv_pipe, comp) and getattr(ig2mv_pipe, comp) is not None:
                    clear_model_attributes(getattr(ig2mv_pipe, comp))
                    setattr(ig2mv_pipe, comp, None)
            
            clear_model_attributes(ig2mv_pipe)
        
        # Clear texture_pipe
        if texture_pipe is not None:
            try:
                texture_pipe.to('cpu')
            except:
                pass
            
            components = ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
            for comp in components:
                if hasattr(texture_pipe, comp) and getattr(texture_pipe, comp) is not None:
                    clear_model_attributes(getattr(texture_pipe, comp))
                    setattr(texture_pipe, comp, None)
            
            clear_model_attributes(texture_pipe)
        
        # Clear references
        ig2mv_pipe = None
        texture_pipe = None
        models_loaded["mv_adapter"] = False
        
        # Aggressive cleanup
        aggressive_cleanup()
        
        print(f"Memory after unloading:\n{get_memory_info()}")
        print("MV-Adapter models unloaded and memory freed.")

def cleanup_models():
    """Unload all models and free maximum memory"""
    print("Starting comprehensive cleanup...")
    print(f"Memory before cleanup:\n{get_memory_info()}")
    
    unload_grounding_sam()
    unload_midi_model()
    unload_mv_adapter()
    
    # Final aggressive cleanup
    aggressive_cleanup()
    
    print(f"Memory after cleanup:\n{get_memory_info()}")
    print("All models unloaded and memory freed.")

@torch.no_grad()
# @torch.autocast(device_type=DEVICE, dtype=torch.float16)
def run_segmentation(
    image_prompts: Any,
    seg_mode: str,
    text_labels: Optional[str] = None,
    polygon_refinement: bool = True,
    detect_threshold: float = 0.3,
) -> Image.Image:
    # Load models on demand
    load_grounding_sam()
    
    rgb_image = image_prompts["image"].convert("RGB")

    segment_kwargs = {}
    if seg_mode == "box":
        # pre-process the layers and get the xyxy boxes of each layer
        if len(image_prompts["points"]) == 0:
            gr.Warning("Please draw bounding boxes for each instance on the image.")
            return None

        boxes = [
            [
                [int(box[0]), int(box[1]), int(box[3]), int(box[4])]
                for box in image_prompts["points"]
            ]
        ]

        if len(boxes) == 0 or any(len(box) == 0 for box in boxes):
            gr.Warning("Please draw bounding boxes for each instance on the image.")
            return None

        segment_kwargs["boxes"] = [boxes]
    else:
        if text_labels is None or text_labels == "" or len(text_labels.split(",")) == 0:
            gr.Warning("Please enter text labels separated by commas.")
            return None

        text_labels = text_labels.split(",")
        detections = detect(object_detector, rgb_image, text_labels, detect_threshold)
        segment_kwargs["detection_results"] = detections

    # run the segmentation
    detections = segment(
        sam_processor,
        sam_segmentator,
        rgb_image,
        polygon_refinement=polygon_refinement,
        **segment_kwargs,
    )
    seg_map_pil = plot_segmentation(rgb_image, detections)

    torch.cuda.empty_cache()

    return seg_map_pil


@torch.no_grad()
def run_generation(
    rgb_image: Any,
    seg_image: Union[str, Image.Image],
    seed: int,
    randomize_seed: bool = False,
    num_inference_steps: int = 35,
    guidance_scale: float = 7.0,
    do_image_padding: bool = False,
):
    # Load MIDI model on demand
    load_midi_model()
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    if not isinstance(rgb_image, Image.Image) and "image" in rgb_image:
        rgb_image = rgb_image["image"]

    # Ensure input images are in the correct format
    if isinstance(rgb_image, Image.Image):
        rgb_image = rgb_image.convert("RGB")
    
    if isinstance(seg_image, Image.Image):
        seg_image = seg_image.convert("RGB")
    
    # Use autocast for mixed precision
    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        scene = run_midi(
            pipe,
            rgb_image,
            seg_image,
            seed,
            num_inference_steps,
            guidance_scale,
            do_image_padding,
        )

    # create uuid for the output file
    output_path = os.path.join(TMP_DIR, f"midi3d_{uuid.uuid4()}.glb")
    scene.export(output_path)

    torch.cuda.empty_cache()

    return output_path, output_path, seed


@torch.no_grad()
def apply_texture(scene_path: str, rgb_image: Any, seg_image: Any, seed: int):
    # Load MV-Adapter models on demand
    load_mv_adapter()
    
    if not isinstance(rgb_image, Image.Image) and "image" in rgb_image:
        rgb_image = rgb_image["image"]

    scene = trimesh.load(scene_path, process=False)
    print(f"Loaded scene with {len(scene.geometry)} meshes")

    # create a tmp dir
    tmp_dir = os.path.join(TMP_DIR, f"textured_{uuid.uuid4()}")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Created temporary directory: {tmp_dir}")

    print("Starting texture generation process...")
    textured_scene = run_i2tex(
        ig2mv_pipe,
        texture_pipe,
        scene,
        rgb_image,
        seg_image,
        seed,
        output_dir=tmp_dir,
    )
    print(
        f"Texture generation completed. Final scene has {len(textured_scene.geometry)} meshes"
    )

    output_path = os.path.join(tmp_dir, "textured_scene.glb")
    textured_scene.export(output_path)
    print(f"Exported textured scene to {output_path}")

    torch.cuda.empty_cache()

    return output_path, output_path, seed


# Demo
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                image_prompts = ImagePrompter(label="Input Image", type="pil")
                seg_image = gr.Image(
                    label="Segmentation Result", type="pil", format="png"
                )

            with gr.Accordion("Segmentation Settings", open=True):
                segmentation_mode = gr.Dropdown(
                    ["box", "label"],
                    value="box",
                    label="Segmentation Mode",
                    info="Box: Draw bounding boxes on the image to generate the segmentation result.\nLabel: Use text labels to segment the image.",
                )
                text_labels = gr.Textbox(
                    label="Text Labels",
                    value="",
                    placeholder="Enter text labels separated by commas if label mode is selected",
                )
                polygon_refinement = gr.Checkbox(label="Polygon Refinement", value=True)
            seg_button = gr.Button("Run Segmentation")

            with gr.Accordion("Generation Settings", open=False):
                do_image_padding = gr.Checkbox(label="Do image padding", value=False)
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=35,
                )
                guidance_scale = gr.Slider(
                    label="CFG scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=7.0,
                )
                
                # Add chunk size control
                chunk_size = gr.Slider(
                    label="Weight Chunk Size (MB)",
                    minimum=100,
                    maximum=1000,
                    step=50,
                    value=CHUNK_SIZE_MB,
                    info="Size of each weight chunk for memory-efficient loading"
                )
                
            gen_button = gr.Button("Run Generation", variant="primary")
            tex_button = gr.Button("Apply Texture", interactive=False)
            
            # Add memory management buttons
            with gr.Accordion("Memory Management", open=True):
                with gr.Row():
                    unload_seg_btn = gr.Button("Unload Segmentation", size="sm")
                    unload_gen_btn = gr.Button("Unload Generation", size="sm")
                    unload_tex_btn = gr.Button("Unload Texture", size="sm")
                unload_all_btn = gr.Button("Unload All Models", variant="stop", size="sm")
                memory_status = gr.Textbox(label="Model Status", interactive=False, value="All models unloaded")
                memory_info = gr.Textbox(label="Memory Info", interactive=False, value=get_memory_info(), lines=3)
                refresh_memory_btn = gr.Button("Refresh Memory Info", size="sm")
                force_gc_btn = gr.Button("Force Garbage Collection", size="sm")

        with gr.Column():
            model_output = gr.Model3D(label="Generated GLB", interactive=False)
            download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
            textured_model_output = gr.Model3D(label="Textured GLB", interactive=False)
            download_textured_glb = gr.DownloadButton(
                label="Download Textured GLB", interactive=False
            )

    with gr.Row():
        gr.Examples(
            examples=EXAMPLES,
            fn=run_generation,
            inputs=[image_prompts, seg_image, seed, randomize_seed, do_image_padding],
            outputs=[model_output, download_glb, seed],
            cache_examples=False,
        )

    def update_memory_status():
        status = []
        if models_loaded["grounding_sam"]:
            status.append("Segmentation: Loaded ✓")
        else:
            status.append("Segmentation: Unloaded")
        if models_loaded["midi"]:
            status.append("Generation: Loaded ✓")
        else:
            status.append("Generation: Unloaded")
        if models_loaded["mv_adapter"]:
            status.append("Texture: Loaded ✓")
        else:
            status.append("Texture: Unloaded")
        return "\n".join(status), get_memory_info()

    def update_chunk_size(new_size):
        global CHUNK_SIZE_MB
        CHUNK_SIZE_MB = new_size
        return f"Chunk size updated to {new_size}MB"

    seg_button.click(
        run_segmentation,
        inputs=[
            image_prompts,
            segmentation_mode,
            text_labels,
            polygon_refinement,
        ],
        outputs=[seg_image],
    ).then(lambda: gr.Button(interactive=True), outputs=[gen_button]).then(
        update_memory_status, outputs=[memory_status, memory_info]
    )

    gen_button.click(
        run_generation,
        inputs=[
            image_prompts,
            seg_image,
            seed,
            randomize_seed,
            num_inference_steps,
            guidance_scale,
            do_image_padding,
        ],
        outputs=[model_output, download_glb, seed],
    ).then(lambda: gr.Button(interactive=True), outputs=[download_glb]).then(
        lambda: gr.Button(interactive=True), outputs=[tex_button]
    ).then(update_memory_status, outputs=[memory_status, memory_info])

    tex_button.click(
        apply_texture,
        inputs=[model_output, image_prompts, seg_image, seed],
        outputs=[textured_model_output, download_textured_glb, seed],
    ).then(lambda: gr.Button(interactive=True), outputs=[download_textured_glb]).then(
        update_memory_status, outputs=[memory_status, memory_info]
    )
    
    # Memory management button handlers
    unload_seg_btn.click(
        unload_grounding_sam,
        outputs=[],
    ).then(update_memory_status, outputs=[memory_status, memory_info])
    
    unload_gen_btn.click(
        unload_midi_model,
        outputs=[],
    ).then(update_memory_status, outputs=[memory_status, memory_info])
    
    unload_tex_btn.click(
        unload_mv_adapter,
        outputs=[],
    ).then(update_memory_status, outputs=[memory_status, memory_info])
    
    unload_all_btn.click(
        cleanup_models,
        outputs=[],
    ).then(update_memory_status, outputs=[memory_status, memory_info])
    
    refresh_memory_btn.click(
        lambda: get_memory_info(),
        outputs=[memory_info]
    )
    
    force_gc_btn.click(
        aggressive_cleanup,
        outputs=[],
    ).then(lambda: get_memory_info(), outputs=[memory_info])
    
    chunk_size.change(
        update_chunk_size,
        inputs=[chunk_size],
        outputs=[gr.Textbox(visible=False)]
    )

demo.launch()
