import os
import uuid
import time
import json
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import gradio_client as grc
import io
from PIL import Image
import base64

# Constants
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
GRADIO_URL = "http://localhost:7860"  # Gradio服务器地址

# Ensure tmp directory exists
os.makedirs(TMP_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="MIDI-3D API", description="API for 3D scene reconstruction via Gradio client")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task storage
task_statuses: Dict[str, Dict] = {}

# Request/Response models
class ProcessRequest(BaseModel):
    seg_mode: str = "box"
    boxes_json: Optional[str] = None  # JSON string
    labels: Optional[str] = None
    polygon_refinement: bool = True
    detect_threshold: float = 0.3
    seed: int = 42
    randomize_seed: bool = True
    num_inference_steps: int = 35
    guidance_scale: float = 7.0
    do_image_padding: bool = False

class ProcessResponse(BaseModel):
    task_id: str
    status_url: str

class ProcessStatus(BaseModel):
    status: str
    message: str
    progress: Optional[float] = None
    model_url: Optional[str] = None

# Gradio client wrapper
class GradioMidiClient:
    def __init__(self, gradio_url: str = GRADIO_URL):
        try:
            self.client = grc.Client(gradio_url)
            self.connected = True
        except Exception as e:
            print(f"Failed to connect to Gradio: {e}")
            self.connected = False
    
    def upload_image(self, image_path: str) -> str:
        """Upload image to Gradio and return the file hash"""
        if not self.connected:
            raise Exception("Gradio client not connected")
        
        # For now, return the local path - Gradio will handle it
        return image_path
    
    def run_segmentation(self, image_path: str, seg_mode: str, boxes_json: Optional[str] = None, 
                        labels: Optional[str] = None, polygon_refinement: bool = True,
                        detect_threshold: float = 0.3) -> str:
        """Run segmentation via Gradio"""
        if not self.connected:
            raise Exception("Gradio client not connected")
        
        # For now, simulate segmentation - replace with actual Gradio call
        seg_path = image_path.replace('.jpg', '_seg.png')
        
        # Create a dummy segmentation result (just copy the original)
        from PIL import Image
        img = Image.open(image_path)
        img.save(seg_path)
        
        return seg_path
    
    def run_generation(self, image_path: str, seg_image_path: str, seed: int, 
                      num_inference_steps: int = 35, guidance_scale: float = 7.0,
                      do_image_padding: bool = False) -> str:
        """Run 3D generation via Gradio"""
        if not self.connected:
            raise Exception("Gradio client not connected")
        
        # For now, simulate generation - replace with actual Gradio call
        import uuid
        model_path = os.path.join(TMP_DIR, f"midi3d_{uuid.uuid4()}.glb")
        
        # Create a dummy GLB file
        with open(model_path, 'wb') as f:
            f.write(b'dummy glb content')
        
        return model_path
    
    def apply_texture(self, model_path: str, image_path: str, seg_image_path: str, seed: int) -> str:
        """Apply texture via Gradio"""
        if not self.connected:
            raise Exception("Gradio client not connected")
        
        # For now, simulate texturing - replace with actual Gradio call
        import uuid
        textured_path = os.path.join(TMP_DIR, f"textured_{uuid.uuid4()}.glb")
        
        # Create a dummy textured GLB file
        with open(textured_path, 'wb') as f:
            f.write(b'dummy textured glb content')
        
        return textured_path

# Task management
def update_task_status(task_id: str, status: str, message: str, progress: float = None, model_url: str = None):
    """Update the status of a task"""
    task_statuses[task_id] = {
        "status": status,
        "message": message,
        "progress": progress,
        "model_url": model_url,
        "timestamp": time.time()
    }

# Background task
def process_image_to_3d(task_id: str, image_path: str, request: ProcessRequest):
    """Process an image to generate a 3D model with textures via Gradio"""
    try:
        # Initialize Gradio client
        client = GradioMidiClient()
        
        # Update status: Starting
        update_task_status(task_id, "processing", "Starting 3D reconstruction process...", 0.05)

        # Run segmentation
        update_task_status(task_id, "processing", "Running segmentation...", 0.2)
        seg_image_path = client.run_segmentation(
            image_path,
            request.seg_mode,
            request.boxes_json,
            request.labels,
            request.polygon_refinement,
            request.detect_threshold,
        )

        # Generate 3D scene
        update_task_status(task_id, "processing", "Generating 3D scene...", 0.5)
        seed = request.seed if not request.randomize_seed else uuid.uuid4().int % (2**32)
        model_path = client.run_generation(
            image_path,
            seg_image_path,
            seed,
            request.num_inference_steps,
            request.guidance_scale,
            request.do_image_padding,
        )

        # Apply textures
        update_task_status(task_id, "processing", "Applying textures to 3D model...", 0.8)
        final_model_path = client.apply_texture(model_path, image_path, seg_image_path, seed)

        # Update status: Complete
        update_task_status(task_id, "completed", "3D model with textures generated successfully!", 1.0, f"/download/{task_id}")

    except Exception as e:
        # Update status: Error
        update_task_status(task_id, "error", f"Error during processing: {str(e)}")
        print(f"Error processing task {task_id}: {str(e)}")

# API Routes
@app.get("/")
async def root():
    """Root endpoint to check API status"""
    return {"message": "MIDI-3D API is running", "status": "active"}

@app.post("/process", response_model=ProcessResponse)
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    seg_mode: str = Query("box"),
    boxes_json: Optional[str] = Query(None),
    labels: Optional[str] = Query(None),
    polygon_refinement: bool = Query(True),
    detect_threshold: float = Query(0.3),
    seed: int = Query(42),
    randomize_seed: bool = Query(True),
    num_inference_steps: int = Query(35),
    guidance_scale: float = Query(7.0),
    do_image_padding: bool = Query(False)
):
    """Process an image to generate a 3D model with textures"""
    print(f"Received request with seg_mode: {seg_mode}")
    print(f"boxes_json: {boxes_json}")
    
    # Check if the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    # Validate segmentation mode
    if seg_mode not in ["box", "label"]:
        raise HTTPException(status_code=400, detail="seg_mode must be either 'box' or 'label'")
    
    # Parse boxes if provided
    parsed_boxes = None
    if boxes_json:
        try:
            parsed_boxes = json.loads(boxes_json)
            print(f"Parsed boxes: {parsed_boxes}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for boxes_json")
    
    # For box mode, boxes are optional - Gradio will handle the error if needed
    if seg_mode == "box" and not parsed_boxes:
        print("Warning: No boxes provided for box mode, proceeding anyway")
    
    # Validate labels if label mode is selected
    if seg_mode == "label" and (not labels or labels == ""):
        raise HTTPException(status_code=400, detail="labels are required when seg_mode is 'label'")

    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Save the uploaded image
    image_path = os.path.join(TMP_DIR, f"{task_id}_input.png")
    with open(image_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    print(f"Saved image to: {image_path}")

    # Create request object
    request = ProcessRequest(
        seg_mode=seg_mode,
        boxes_json=boxes_json,  # Keep as JSON string
        labels=labels,
        polygon_refinement=polygon_refinement,
        detect_threshold=detect_threshold,
        seed=seed,
        randomize_seed=randomize_seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        do_image_padding=do_image_padding
    )

    # Initialize task status
    update_task_status(task_id, "queued", "Task queued for processing")

    # Add the processing task to background tasks
    background_tasks.add_task(process_image_to_3d, task_id, image_path, request)

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

    model_path = os.path.join(TMP_DIR, f"midi3d_{task_id}.glb")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=f"midi3d_{task_id}.glb"
    )

@app.get("/gradio/status")
async def get_gradio_status():
    """Check if Gradio server is accessible"""
    try:
        client = GradioMidiClient()
        if client.connected:
            return {"status": "connected", "url": GRADIO_URL}
        else:
            return {"status": "disconnected", "error": "Connection failed", "url": GRADIO_URL}
    except Exception as e:
        return {"status": "disconnected", "error": str(e), "url": GRADIO_URL}

# Start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
