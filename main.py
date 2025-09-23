import os
import shutil
import uuid
from typing import List, Optional
from crew import ProposalCrew
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





# ------------------------------
# Define JSON Schema for Output
# ------------------------------



@app.get("/")
def health_check():
    return {"status": "ok", "message": "Proposal API is running!"}




UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/proposal")
async def generate_proposal(files: List[UploadFile] = File(...)):
    """
    Accept PDFs, save them, and return generated proposal.
    """

    # Save uploaded PDFs
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    
    result = ProposalCrew().crew().kickoff()

    if result.json_dict:
        data = result.json_dict
    elif result.pydantic:
        data = result.pydantic.dict()
    else:
        data= result.raw

    try:
        # Remove vector DBs (adjust paths as per your setup)
        if os.path.exists("db"):
            shutil.rmtree("db")
        if os.path.exists("db1"):
            shutil.rmtree("db1")

        # Remove uploaded PDFs
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)  # recreate empty uploads
    except Exception as e:
        print(f"Cleanup failed: {e}")
    

    return data


