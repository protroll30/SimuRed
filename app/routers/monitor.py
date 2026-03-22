from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Import the services we built
from app.services.mutator import PromptMutator
from app.services.llm_client import LLMClient
from app.services.evaluator import LogicEvaluator

# Define the router with metadata for the /docs page
router = APIRouter(
    prefix="/monitor",
    tags=["SimuRed Monitoring"]
)

# Initialize services globally so they persist across requests
mutator = PromptMutator()
llm_client = LLMClient()
evaluator = LogicEvaluator()

# Define the shape of the incoming data
class PromptRequest(BaseModel):
    prompt: str

@router.post("/test-logic")
async def test_logic(request: PromptRequest):
    """
    The core SimuRed endpoint: 
    Mutates a prompt, queries Gemini, and evaluates logical drift.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        # 1. Generate the 'Attacks' (Original, Typo, Semantic)
        # This uses NLPAug under the hood.
        attack_dict = mutator.generate_attacks(request.prompt)
        
        # 2. Get AI Responses for all versions in parallel
        # This uses litellm + async to stay fast.
        responses = await llm_client.get_responses(attack_dict)
        
        # 3. Calculate Logical Drift (The Stats Flex)
        # Compares attack responses against the original baseline.
        drift_report = evaluator.evaluate_drift(responses)
        
        # 4. Package the results into a clean list
        simulations = []
        for attack_type in attack_dict:
            # We use .get() for drift_metrics because the 'original' 
            # version won't have a drift score (it IS the baseline).
            simulations.append({
                "type": attack_type,
                "input": attack_dict[attack_type],
                "output": responses[attack_type],
                "drift_metrics": drift_report.get(attack_type, "Baseline / Original")
            })
        
        return {
            "status": "success",
            "model_tested": "gemini-2.5-flash",
            "simulations": simulations
        }

    except Exception as e:
        # Professional error handling so the API doesn't just 'die'
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")