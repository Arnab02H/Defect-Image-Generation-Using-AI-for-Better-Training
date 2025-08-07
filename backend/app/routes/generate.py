from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse,FileResponse
from pathlib import Path
import shutil
import logging
from app.utils.image_processing import load_images_from_zip
from app.utils.training import train_gan, generate_images
from app.utils.file_handling import save_images_to_zip, get_sample_images
from io import BytesIO
import os
router = APIRouter()

UPLOAD_DIR = Path("uploads")
GENERATED_DIR = Path("generated")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@router.post("/generate")
async def generate(file: UploadFile = File(...), num_images: int = Form(...)):
    logger.info(f"Received file: {file.filename}, num_images: {num_images}")
    
    # Validate inputs
    if not file.filename.endswith('.zip'):
        logger.error("Invalid file format: not a ZIP file")
        raise HTTPException(status_code=400, detail="File must be a ZIP file")
    if num_images < 1 or num_images > 100:
        logger.error(f"Invalid number of images: {num_images}")
        raise HTTPException(status_code=400, detail="Number of images must be between 1 and 100")
    if file.size > 10* 1024 * 1024:  # 100MB limit
        logger.error("File too large")
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

    # Save uploaded ZIP
    zip_path = UPLOAD_DIR / file.filename
    try:
        with open(zip_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"Saved uploaded file to {zip_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Load and preprocess images
    try:
        images = load_images_from_zip(zip_path)
        if not images.size(0):
            logger.error("No valid images found in ZIP file")
            raise HTTPException(status_code=400, detail="No valid PNG/JPEG images found in ZIP file")
        logger.info(f"Loaded {images.size(0)} images from ZIP")
    except Exception as e:
        logger.error(f"Image loading failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process images")

    # Train GAN
    try:
        generator = train_gan(images)
        logger.info("GAN training completed")
    except Exception as e:
        logger.error(f"GAN training failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to train GAN")

    # Generate images
    try:
        generated_imgs = generate_images(images,generator, num_images)
        logger.info(f"Generated {num_images} images")
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        raise HTTPException(status_code=500)

    # Save generated images to ZIP
    output_zip_path = GENERATED_DIR / "generated_images.zip"
    try:
        save_images_to_zip(generated_imgs, output_zip_path)
        logger.info(f"Saved generated images to {output_zip_path}")
    except Exception as e:
        logger.error(f"Failed to save generated images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save generated images")

    # # Clean up
    try:
        shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(exist_ok=True)
        logger.info("Cleaned up uploads directory")
    except Exception as e:
        logger.warning(f"Failed to clean up uploads: {str(e)}")

    # Return ZIP file
    if not output_zip_path.exists():
        logger.error("Generated ZIP file not found")
        raise HTTPException(status_code=500, detail="Generated ZIP file not found")
    #Read the file into memory
    zip_bytes=BytesIO()
    with open(output_zip_path,'rb') as f:
        zip_bytes.write(f.read())
    zip_bytes.seek(0)
    return StreamingResponse(
        zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=generated_images.zip"}
    )
### Showing sample images 


@router.get("/sample-images")
async def sample_images():
    output_zip_path = UPLOAD_DIR 
    if not output_zip_path.exists():
        logger.error("No generated images available")
        raise HTTPException(status_code=404, detail="No generated images available")
    
    try:
        sample_images = get_sample_images(output_zip_path, num_samples=5)
        if not sample_images:
            logger.error("No valid sample images found")
            raise HTTPException(status_code=400, detail="No valid sample images found")
        logger.info(f"Returning {len(sample_images)} sample images")
        return {"images": sample_images}
    except Exception as e:
        logger.error(f"Failed to fetch sample images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch sample images")
#### showing generated images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZIP_PATH = os.path.join(BASE_DIR, "generated", "generated_images.zip")
@router.get("/generated-images")
async def generated_images():
    pth= ZIP_PATH
    if not pth.exists():
        logger.error("No generated images available")
        raise HTTPException(status_code=404, detail="No generated images available")
    try:
        generate_images = get_sample_images(pth, num_samples=5)
        if not sample_images:
            logger.error("No valid sample images found")
            raise HTTPException(status_code=400, detail="No valid sample images found")
        logger.info(f"Returning {len(generate_images)} sample images")
        return {"images": generate_images}
    except Exception as e:
        logger.error(f"Failed to fetch sample images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch sample images")
## Download images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZIP_PATH = os.path.join(BASE_DIR, "generated", "generated_images.zip")
@router.get("/download_zip")
async def download_zip():
    if not os.path.isfile(ZIP_PATH):
        print("File does not exist!")
        raise HTTPException(status_code=404, detail=f"ZIP file not found at {ZIP_PATH}")
    print("File found, serving...")
    return FileResponse(
        path=ZIP_PATH,
        filename="generated_images.zip",
        media_type="application/zip"
    )