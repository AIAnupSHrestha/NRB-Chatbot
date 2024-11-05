from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional
import uvicorn
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Import the RAG pipeline
from main import SimpleRAGPipeline

app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation using PDFs",
    version="1.0.0"
)

# Initialize the RAG pipeline
rag = SimpleRAGPipeline()

class QueryRequest(BaseModel):
    question: str

class PromptTemplate(BaseModel):
    template: str

class ResponseModel(BaseModel):
    query_response: str
    data: list = []
    type: str = "normal_message"
    data_fetch_status: str

def create_response(response: str, status: str = "success") -> dict:
    return {
        "query_response": response,
        "data": [],
        "type": "normal_message",
        "data_fetch_status": status
    }

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = 500,
    chunk_overlap: Optional[int] = 50
):
    """
    Upload and process a PDF file.
    """
    if not file.filename.endswith('.pdf'):
        return JSONResponse(
            content=create_response("File must be a PDF", "failed"),
            status_code=400
        )
    
    try:
        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        rag.upload_pdf(
            pdf_path=tmp_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        os.unlink(tmp_path)
        
        return JSONResponse(
            content=create_response(f"PDF {file.filename} successfully processed"),
            status_code=200
        )
    
    except Exception as e:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        return JSONResponse(
            content=create_response(str(e), "failed"),
            status_code=500
        )

@app.post("/query")
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    """
    try:
        # Get the answer and retrieved passages
        retrieved_text = rag.query(request.question)  # Assuming your RAG pipeline has this method

        
        # Construct the response with the retrieved text
        return {
            "query_response": retrieved_text,  # The generated answer
            "data": [],    # List of retrieved passages
            "type": "normal_message",
            "data_fetch_status": "success"
        }
    except Exception as e:
        return {
            "query_response": str(e),
            "data": [],
            "type": "normal_message",
            "data_fetch_status": "failed"
        }

@app.post("/set-prompt")
async def set_prompt(prompt: PromptTemplate):
    """
    Set a custom prompt template.
    """
    try:
        rag.set_custom_prompt(prompt.template)
        return create_response("Prompt template successfully updated")
    except ValueError as e:
        return create_response(str(e), "failed")
    except Exception as e:
        return create_response(str(e), "failed")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return create_response("healthy")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)