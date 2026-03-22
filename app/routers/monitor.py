from fastapi import APIRouter
from pydantic import BaseModel
from app.services.mutator import PromptMutator
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/monitor", tags=["monitoring"])

# Initialize the engines
mutator = PromptMutator()
llm_client = LLMClient()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/test-logic")
async def test_logic(request: PromptRequest):
    # Generate the 'Attacks' (Original, Typo, Semantic)
    attack_dict = mutator.generate_attacks(request.prompt)
    
    # Get AI Responses for all 3 versions at once
    responses = await llm_client.get_responses(attack_dict)
    
    # Format the result for the user
    results = []
    for attack_type in attack_dict:
        results.append({
            "type": attack_type,
            "input": attack_dict[attack_type],
            "output": responses[attack_type]
        })
    
    return {
        "status": "success",
        "simulations": results
    }