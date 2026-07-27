from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from controllers.profile_controller import profile_controller

router = APIRouter(prefix="/api/profiles", tags=["Profiles"])


@router.get("")
@router.get("/")
async def list_profiles():
    """Returns all enrolled person profiles with thumbnail images and embedding statistics."""
    profiles = await profile_controller.get_all_profiles()
    return {"status": "success", "count": len(profiles), "profiles": profiles}


@router.post("/create")
async def create_profile(profile_id: str = Form(...), name: Optional[str] = Form(None)):
    """Creates a new person profile folder and database entry."""
    result = await profile_controller.create_profile(profile_id, name)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=str(result.get("message")))
    return result


@router.post("/enroll-image")
async def enroll_image(
    profile_id: str = Form(...),
    name: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    """
    Uploads a face photo for a person:
    - Enforces strict single-face frame validation (rejects 0 faces or >1 face).
    - Deduplicates 512D ArcFace embeddings (skips FAISS addition if similarity > 0.95).
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")

    file_bytes = await file.read()
    result = await profile_controller.enroll_face_image(
        profile_id=profile_id,
        file_bytes=file_bytes,
        filename=file.filename,
        name=name
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)

    return result


@router.delete("/{profile_id}")
async def delete_profile(profile_id: str):
    """Deletes a person profile, removes images, and rebuilds the FAISS vector index."""
    result = await profile_controller.delete_profile(profile_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=str(result.get("message")))
    return result
