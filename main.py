import os
import uuid
import shutil
import subprocess
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from crew import even_list
from crew import ProposalCrew  # your crew implementation

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global state
is_running: bool = False
final_result: Optional[dict] = None


def run_proposal(files: List[str]):
    """
    Runs crew in the background and updates global state.
    """
    global is_running, final_result

    try:
        
        result = ProposalCrew().crew().kickoff()

        if result.json_dict:
            final_result = result.json_dict
        elif result.pydantic:
            final_result = result.pydantic.dict()
        else:
            final_result = {"raw": str(result.raw)}

    except Exception as e:
        final_result = {"error": str(e)}
    finally:
        # Cleanup after run
        try:
            #if os.path.exists("db"):
            #    shutil.rmtree("db")
            #if os.path.exists("db1"):
            #      shutil.rmtree("db1")
            if os.path.exists(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR, exist_ok=True)
        except Exception as ce:
            print(f"Cleanup failed: {ce}")
        even_list.clear()
        is_running = False


@app.post("/proposal")
async def generate_proposal(
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = File(None),
    db_action: Optional[str] = Form(None),  # 'keep' or 'delete'
):
    """
    Accept document files, save them, and run ProposalCrew in background.
    Supported extensions: .pdf, .docx, .xlsx, .pptx
    """
    global is_running, final_result
    if is_running:
        raise HTTPException(status_code=400, detail="Proposal already running.")

    # Check if vector DB already exists
    chroma_db_path = os.path.join("db", "chroma.sqlite3")
    db_exists = os.path.exists(chroma_db_path)

    # If DB exists and no action provided, prompt frontend to decide
    if db_exists and db_action not in {"keep", "delete"}:
        raise HTTPException(
            status_code=409,
            detail={
                "db_exists": True,
                "message": "Vector DB already exists. Provide db_action as 'keep' to use existing DB or 'delete' to rebuild.",
                "allowed_actions": ["keep", "delete"],
            },
        )

    # Handle delete action: remove DB and require files
    if db_exists and db_action == "delete":
        try:
            if os.path.exists("db"):
                shutil.rmtree("db")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete existing DB: {e}")

    # Determine if we need files this run
    require_files = (not db_exists) or (db_exists and db_action == "delete")

    if require_files:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided. Upload .pdf, .docx, .xlsx, or .pptx files.")

        allowed_ext = {".pdf", ".docx", ".xlsx", ".pptx"}

        saved_files = []
        for file in files:
            _, ext = os.path.splitext(file.filename.lower())
            if ext not in allowed_ext:
                raise HTTPException(status_code=400, detail="Only PDF, DOCX, XLSX, PPTX files are allowed.")
            file_id = str(uuid.uuid4())
            file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
    else:
        # Using existing DB; ignore any provided files
        saved_files = []

    is_running = True
    final_result = None

    # Run crew in background (it will build a new DB if needed when queried)
    background_tasks.add_task(run_proposal, saved_files)

    return {
        "message": "Proposal started. Poll /status for updates.",
        "db_used": "existing" if (db_exists and db_action == "keep") else ("rebuilt" if db_exists else "new"),
    }


@app.get("/status")
def get_status():
    """
    Return current even_list, running status, and final result (if ready).
    """
    return {
        "running": is_running,
        "even_list": even_list,
        "result": final_result,
    }


@app.post("/reset")
def reset_system():
    """
    Reset the system by deleting db1 directory and clearing uploads folder.
    Also reloads the frontend on port 3000.
    """
    global is_running, final_result
    
    try:
        # Stop any running proposal
        is_running = False
        final_result = None
        even_list.clear()
        
        # Delete db1 directory if it exists
        if os.path.exists("db1"):
            shutil.rmtree("db1")
            print("Deleted db1 directory")
        if os.path.exists("db"):
            shutil.rmtree("db")
            print("Deleted db directory")
        
        # Clear uploads directory
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            print("Cleared uploads directory")
        
       
        return {"message": "System reset successfully. Database cleared, uploads cleared, and frontend reloaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
