from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(
    prefix="/monitor",
    tags=["monitoring"]
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/test-logic")
async def test_logic(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    return {
        "original_prompt": request.prompt,
        "status": "received"
        "message": "SimuRed simulation initiated"
    }