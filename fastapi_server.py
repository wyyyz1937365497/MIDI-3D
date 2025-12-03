import json
import os
import uuid
import time
import gc
import psutil
import io
import base64
import numpy as np
import torch
import trimesh
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any

# Import MIDI-3D components
from midi.pipelines.pipeline_midi import MIDIPipeline
from scripts.grounding_sam import detect, plot_segmentation, prepare_model, segment
from scripts.image_to_textured_scene import (
    prepare_ig2mv_pipeline,
    prepare_texture_pipeline,
    run_i2tex,
)
from scripts.inference_midi import run_midi
from huggingface_hub import snapshot_download

def initialize_globals():
    global object_detector, sam_processor, sam_segmentator, pipe, ig2mv_pipe, texture_pipe
    object_detector = None
    sam_processor = None
    sam_segmentator = None
    pipe = None
    ig2mv_pipe = None
    texture_pipe = None

initialize_globals()

# Constants
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "VAST-AI/MIDI-3D"
CHUNK_SIZE_MB = 500

# Ensure tmp directory exists
os.makedirs(TMP_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="MIDI-3D API", description="API for 3D scene reconstruction from images")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model containers
object_detector = None
sam_processor = None
sam_segmentator = None
pipe = None
ig2mv_pipe = None
texture_pipe = None

# Model loading status
models_loaded = {
    "grounding_sam": False,
    "midi": False,
    "mv_adapter": False,
}

# Response models
class ProcessStatus(BaseModel):
    status: str
    message: str
    progress: Optional[float] = None
    model_url: Optional[str] = None

class ProcessResponse(BaseModel):
    task_id: str
    status_url: str

# Request models
class BoundingBox(BaseModel):
    x1: int  # 左上角x坐标
    y1: int  # 左上角y坐标
    x2: int  # 右下角x坐标
    y2: int  # 右下角y坐标

class SegmentationRequest(BaseModel):
    mode: str = "box"  # "box" 或 "label"
    boxes: Optional[List[BoundingBox]] = None  # 矩形框列表
    labels: Optional[str] = None  # 文本标签，逗号分隔
    polygon_refinement: bool = True
    detect_threshold: float = 0.3

# In-memory storage for task statuses (in production, use a database)
task_statuses = {}

class ChunkedWeightLoader:
    """Manages chunked loading of model weights to minimize RAM usage"""

    def __init__(self, chunk_size_mb: int = CHUNK_SIZE_MB):
        self.chunk_size_mb = chunk_size_mb
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024

    def estimate_tensor_size(self, tensor: torch.Tensor) -> int:
        """Estimate the size of a tensor in bytes"""
        return tensor.numel() * tensor.element_size()

    def split_state_dict(self, state_dict: dict) -> list:
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

    def load_model_chunked(self, model, state_dict_path: str, device: str = DEVICE, dtype: torch.dtype = DTYPE) -> None:
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
    info = {}

    # GPU Memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        info["gpu"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
        }
    else:
        info["gpu"] = {"status": "Not Available"}

    # System RAM
    process = psutil.Process(os.getpid())
    ram_info = process.memory_info()
    ram_used = ram_info.rss / 1024**3  # GB
    ram_percent = process.memory_percent()
    system_ram = psutil.virtual_memory()
    system_ram_used = system_ram.used / 1024**3
    system_ram_total = system_ram.total / 1024**3

    info["ram"] = {
        "process_used_gb": round(ram_used, 2),
        "process_percent": round(ram_percent, 1),
        "system_used_gb": round(system_ram_used, 1),
        "system_total_gb": round(system_ram_total, 1),
        "system_percent": system_ram.percent,
    }

    return info

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
    """Load Grounding SAM models on demand - Gradio style"""
    global object_detector, sam_processor, sam_segmentator, models_loaded

    if not models_loaded["grounding_sam"]:
        print("Loading Grounding SAM models...")
        print(f"Memory before loading:\n{get_memory_info()}")
        
        try:
            # 直接使用Gradio的加载方式，不进行复杂的GPU移动
            object_detector, sam_processor, sam_segmentator = prepare_model(
                device=DEVICE,
                detector_id="IDEA-Research/grounding-dino-tiny",
                segmenter_id="facebook/sam-vit-base",
            )
            
            # 验证加载结果
            if object_detector is None or sam_processor is None or sam_segmentator is None:
                raise RuntimeError("One or more models failed to load (returned None)")
            
            models_loaded["grounding_sam"] = True
            
            print(f"Memory after loading:\n{get_memory_info()}")
            print("Grounding SAM models loaded successfully.")
            
        except Exception as e:
            print(f"Error loading Grounding SAM models: {e}")
            import traceback
            traceback.print_exc()
            
            # 重置状态
            object_detector = None
            sam_processor = None
            sam_segmentator = None
            models_loaded["grounding_sam"] = False
            raise RuntimeError(f"Failed to load Grounding SAM models: {str(e)}")


def load_midi_model():
    """Load MIDI model on demand with chunked loading"""
    global pipe, models_loaded

    if not models_loaded["midi"]:
        print("Loading MIDI model with chunked weight loading...")

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

        print("MIDI model loaded successfully with chunked weights.")

def load_mv_adapter():
    """Load MV-Adapter models on demand"""
    global ig2mv_pipe, texture_pipe, models_loaded

    if not models_loaded["mv_adapter"]:
        print("Loading MV-Adapter models...")

        ig2mv_pipe = prepare_ig2mv_pipeline(device="cuda", dtype=torch.float16)
        texture_pipe = prepare_texture_pipeline(device="cuda", dtype=torch.float16)
        models_loaded["mv_adapter"] = True

        print("MV-Adapter models loaded successfully.")

def cleanup_models():
    """Unload all models and free maximum memory"""
    print("Starting comprehensive cleanup...")

    # Unload Grounding SAM
    if models_loaded["grounding_sam"]:
        if object_detector is not None:
            clear_model_attributes(object_detector)
        if sam_segmentator is not None:
            clear_model_attributes(sam_segmentator)

        object_detector = None
        sam_processor = None
        sam_segmentator = None
        models_loaded["grounding_sam"] = False

    # Unload MIDI
    if models_loaded["midi"] and pipe is not None:
        try:
            pipe.to('cpu')
        except:
            pass

        components = ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler', 
                     'feature_extractor', 'image_encoder', 'safety_checker']

        for comp in components:
            if hasattr(pipe, comp) and getattr(pipe, comp) is not None:
                clear_model_attributes(getattr(pipe, comp))
                setattr(pipe, comp, None)

        clear_model_attributes(pipe)
        pipe = None
        models_loaded["midi"] = False

    # Unload MV-Adapter
    if models_loaded["mv_adapter"]:
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

        ig2mv_pipe = None
        texture_pipe = None
        models_loaded["mv_adapter"] = False

    # Aggressive cleanup
    aggressive_cleanup()

    print("All models unloaded and memory freed.")

def update_task_status(task_id: str, status: str, message: str, progress: float = None, model_url: str = None):
    """Update the status of a task"""
    task_statuses[task_id] = {
        "status": status,
        "message": message,
        "progress": progress,
        "model_url": model_url,
        "timestamp": time.time()
    }

def process_image_to_3d(
    task_id: str, 
    image_path: str, 
    seg_mode: str = "box",
    boxes: Optional[List[Any]] = None,  # Changed to Any to handle both dict and list formats
    labels: Optional[str] = None,
    polygon_refinement: bool = True,
    detect_threshold: float = 0.3
):
    """Process an image to generate a 3D model with textures - Gradio style"""
    global object_detector, sam_processor, sam_segmentator, pipe, ig2mv_pipe, texture_pipe, models_loaded
    
    def cleanup_pipeline(pipeline, components):
        """封装的清理函数，避免重复代码"""
        if pipeline is None:
            return
        try:
            pipeline.to('cpu')
        except:
            pass
        for comp in components:
            if hasattr(pipeline, comp) and getattr(pipeline, comp) is not None:
                clear_model_attributes(getattr(pipeline, comp))
                setattr(pipeline, comp, None)
        clear_model_attributes(pipeline)
    
    try:
        # Update status: Starting
        update_task_status(task_id, "processing", "Starting 3D reconstruction process...", 0.05)

        # Load models as needed
        update_task_status(task_id, "processing", "Loading segmentation models...", 0.1)
        load_grounding_sam()
        
        # 关键验证：确保模型已正确加载
        if sam_processor is None or sam_segmentator is None or object_detector is None:
            error_msg = "Segmentation models are not properly loaded"
            update_task_status(task_id, "error", error_msg)
            raise RuntimeError(error_msg)

        # Load the image
        update_task_status(task_id, "processing", "Loading image...", 0.15)
        rgb_image = Image.open(image_path).convert("RGB")

        # Prepare segmentation parameters - 使用Gradio的逻辑
        segment_kwargs = {}

        if seg_mode == "box":
            # Process bounding boxes (already formatted by the API endpoint)
            if boxes is None or len(boxes) == 0:
                raise ValueError("No bounding boxes provided for box mode")

            # The 'boxes' variable is now already in the correct format from the API endpoint
            segment_kwargs["boxes"] = [boxes]
            update_task_status(task_id, "processing", f"Processing {len(boxes[0])} bounding boxes...", 0.2)
        else:
            # Process text labels
            if labels is None or labels == "":
                raise ValueError("No labels provided for label mode")

            text_labels = labels.split(",")
            update_task_status(task_id, "processing", f"Detecting objects with labels: {', '.join(text_labels)}...", 0.2)
            detections = detect(object_detector, rgb_image, text_labels, detect_threshold)
            segment_kwargs["detection_results"] = detections

        # Run the segmentation - 使用Gradio的torch.no_grad()模式
        update_task_status(task_id, "processing", "Running segmentation...", 0.25)
        
        with torch.no_grad():
            detections = segment(
                sam_processor,
                sam_segmentator,
                rgb_image,
                polygon_refinement=polygon_refinement,
                **segment_kwargs,
            )
            seg_map_pil = plot_segmentation(rgb_image, detections)

        # Save segmentation result temporarily
        seg_path = os.path.join(TMP_DIR, f"{task_id}_seg.png")
        seg_map_pil.save(seg_path)

        # Clean up segmentation models to free memory
        update_task_status(task_id, "processing", "Cleaning up segmentation models...", 0.25)
        if models_loaded["grounding_sam"]:
            try:
                if object_detector is not None:
                    clear_model_attributes(object_detector)
                if sam_segmentator is not None:
                    clear_model_attributes(sam_segmentator)
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")

            object_detector = None
            sam_processor = None
            sam_segmentator = None
            models_loaded["grounding_sam"] = False

            aggressive_cleanup()

        # Load MIDI model
        update_task_status(task_id, "processing", "Loading 3D generation model...", 0.3)
        load_midi_model()

        # Generate 3D scene - 使用Gradio的torch.no_grad()和autocast
        update_task_status(task_id, "processing", "Generating 3D scene...", 0.5)
        
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                scene = run_midi(
                    pipe,
                    rgb_image,
                    seg_map_pil,
                    seed=42,  # Fixed seed for reproducibility
                    num_inference_steps=35,
                    guidance_scale=7.0,
                    do_image_padding=True,
                )

        # Save the 3D scene
        scene_path = os.path.join(TMP_DIR, f"{task_id}_scene.glb")
        scene.export(scene_path)

        # Clean up MIDI model to free memory
        update_task_status(task_id, "processing", "Cleaning up 3D generation model...", 0.6)
        if models_loaded["midi"] and pipe is not None:
            try:
                pipe.to('cpu')
            except:
                pass

            components = ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler', 
                         'feature_extractor', 'image_encoder', 'safety_checker']

            for comp in components:
                if hasattr(pipe, comp) and getattr(pipe, comp) is not None:
                    clear_model_attributes(getattr(pipe, comp))
                    setattr(pipe, comp, None)

            clear_model_attributes(pipe)
            pipe = None
            models_loaded["midi"] = False

            aggressive_cleanup()

        # Load MV-Adapter models
        update_task_status(task_id, "processing", "Loading texture generation models...", 0.7)
        load_mv_adapter()

        # Apply textures - 使用Gradio的torch.no_grad()模式
        update_task_status(task_id, "processing", "Applying textures to 3D model...", 0.8)
        scene = trimesh.load(scene_path, process=False)

        # Create a temporary directory for textured model
        tmp_dir = os.path.join(TMP_DIR, f"textured_{task_id}")
        os.makedirs(tmp_dir, exist_ok=True)

        with torch.no_grad():
            # Generate textured scene
            textured_scene = run_i2tex(
                ig2mv_pipe,
                texture_pipe,
                scene,
                rgb_image,
                seg_map_pil,
                seed=42,  # Fixed seed for reproducibility
                output_dir=tmp_dir,
            )

        # Export the final textured model
        final_model_path = os.path.join(tmp_dir, "textured_scene.glb")
        textured_scene.export(final_model_path)

        # Clean up MV-Adapter models - 修复后的部分
        update_task_status(task_id, "processing", "Cleaning up texture generation models...", 0.9)
        if models_loaded["mv_adapter"]:
            # 清理ig2mv_pipe - 使用正确的变量引用
            if ig2mv_pipe is not None:
                cleanup_pipeline(ig2mv_pipe, ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler'])

            # 清理texture_pipe - 使用正确的变量引用
            if texture_pipe is not None:
                cleanup_pipeline(texture_pipe, ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler'])

            ig2mv_pipe = None
            texture_pipe = None
            models_loaded["mv_adapter"] = False

            aggressive_cleanup()

        # Final cleanup
        update_task_status(task_id, "processing", "Finalizing model...", 0.95)

        # Clean up temporary files
        if os.path.exists(seg_path):
            os.remove(seg_path)
        if os.path.exists(scene_path):
            os.remove(scene_path)

        # Update status: Complete
        update_task_status(task_id, "completed", "3D model with textures generated successfully!", 1.0, f"/download/{task_id}")

    except Exception as e:
        # Update status: Error
        update_task_status(task_id, "error", f"Error during processing: {str(e)}")
        print(f"Error processing task {task_id}: {str(e)}")
        import traceback
        traceback.print_exc()

def get_memory_info_str():
    """Get comprehensive memory usage information as a string"""
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


@app.get("/")
async def root():
    """Root endpoint to check API status"""
    return {"message": "MIDI-3D API is running", "status": "active"}

@app.post("/process", response_model=ProcessResponse)
async def process_image(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    seg_mode: str = Form("box"),
    boxes_json: Optional[str] = Form(None),
    labels: Optional[str] = Form(None),
    polygon_refinement: bool = Form(True),
    detect_threshold: float = Form(0.3)
):
    """Process an uploaded image to generate a 3D model with textures

    Parameters:
    - file: 上传的图片文件
    - seg_mode: 分割模式，"box"（使用矩形框）或 "label"（使用文本标签）
    - boxes_json: 矩形框的JSON字符串，格式为 [{"x1":100,"y1":100,"x2":200,"y2":200}, ...] 或 [[100,100,200,200], ...]
    - labels: 文本标签，用逗号分隔，仅在seg_mode为"label"时使用
    - polygon_refinement: 是否使用多边形优化
    - detect_threshold: 检测阈值，仅在seg_mode为"label"时使用
    """
    # Check if the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    # Validate segmentation mode
    if seg_mode not in ["box", "label"]:
        raise HTTPException(status_code=400, detail="seg_mode must be either 'box' or 'label'")

    # Validate boxes if box mode is selected
    if seg_mode == "box" and boxes_json is None:
        raise HTTPException(status_code=400, detail="boxes_json is required when seg_mode is 'box'")

    # Validate labels if label mode is selected
    if seg_mode == "label" and (labels is None or labels == ""):
        raise HTTPException(status_code=400, detail="labels is required when seg_mode is 'label'")

    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Save the uploaded image
    image_path = os.path.join(TMP_DIR, f"{task_id}_input.png")
    with open(image_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Parse and format boxes if provided（修正部分）
    formatted_boxes = None
    if boxes_json is not None:
        try:
            parsed_boxes = json.loads(boxes_json)
            
            # 验证解析后的数据不为空
            if not parsed_boxes:
                raise ValueError("boxes_json cannot be an empty list")

            # 检查第一个元素的类型以确定格式
            first_element = parsed_boxes[0]
            
            # 情况1: 字典列表格式 [{"x1":..., "y1":..., ...}, ...]
            if isinstance(first_element, dict):
                # 转换为坐标列表并添加三层嵌套以匹配Gradio格式
                coordinate_list = [
                    [int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])]
                    for box in parsed_boxes
                ]
                # 关键修正：添加额外的嵌套层级以匹配Gradio格式
                formatted_boxes = [coordinate_list]
                
            # 情况2: 列表的列表格式 [[x1, y1, x2, y2], ...]
            elif isinstance(first_element, list):
                # 验证每个坐标列表有4个元素
                for box in parsed_boxes:
                    if len(box) != 4:
                        raise ValueError("Each box must contain exactly 4 coordinates [x1, y1, x2, y2]")
                
                # 转换为整数并添加三层嵌套以匹配Gradio格式
                coordinate_list = [
                    [int(coord) for coord in box]
                    for box in parsed_boxes
                ]
                # 关键修正：添加额外的嵌套层级以匹配Gradio格式
                formatted_boxes = [coordinate_list]
                
            else:
                # 不支持的格式
                raise TypeError("boxes_json must be a list of objects or a list of coordinate lists")

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for boxes_json")
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid box format: {str(e)}")

    # Initialize task status
    update_task_status(task_id, "queued", "Task queued for processing")

    # Add the processing task to background tasks（传递格式化后的boxes）
    background_tasks.add_task(
        process_image_to_3d, 
        task_id, 
        image_path, 
        seg_mode, 
        formatted_boxes,  # 使用修正后的三层嵌套格式
        labels, 
        polygon_refinement, 
        detect_threshold
    )

    # Return the task ID and status URL
    return ProcessResponse(
        task_id=task_id,
        status_url=f"/status/{task_id}"
    )

@app.get("/status/{task_id}", response_model=ProcessStatus)
async def get_status(task_id: str):
    """Get the status of a processing task"""
    if task_id not in task_statuses:
        raise HTTPException(status_code=404, detail="Task not found")

    status = task_statuses[task_id]
    return ProcessStatus(
        status=status["status"],
        message=status["message"],
        progress=status.get("progress"),
        model_url=status.get("model_url")
    )

@app.get("/download/{task_id}")
async def download_model(task_id: str):
    """Download the generated 3D model"""
    if task_id not in task_statuses:
        raise HTTPException(status_code=404, detail="Task not found")

    status = task_statuses[task_id]

    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")

    model_path = os.path.join(TMP_DIR, f"textured_{task_id}", "textured_scene.glb")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=f"midi3d_{task_id}.glb"
    )

@app.get("/memory")
async def get_memory():
    """Get memory usage information"""
    return get_memory_info_str()

@app.post("/cleanup")
async def cleanup():
    """Unload all models and free memory"""
    cleanup_models()
    return {"message": "All models unloaded and memory freed"}

# Start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
