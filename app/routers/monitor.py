from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.mutator import PromptMutator
from app.services.llm_client import LLMClient
from app.services.evaluator import LogicEvaluator
from app.services.database import DatabaseService

router = APIRouter(
    prefix="/monitor",
    tags=["SimuRed Monitoring"]
)

mutator = PromptMutator()
llm_client = LLMClient()
evaluator = LogicEvaluator(llm_client)
db = DatabaseService()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/test-logic")
async def test_logic(request: PromptRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        attack_dict = mutator.generate_attacks(request.prompt)
        responses = await llm_client.get_responses(attack_dict)
        drift_report = await evaluator.evaluate_drift(responses)

        simulations = []
        for attack_type in attack_dict:
            sim_data = {
                "type": attack_type,
                "input": attack_dict[attack_type],
                "output": responses[attack_type],
                "drift_metrics": drift_report.get(attack_type, {"similarity_score": 1.0, "is_stable": True})
            }
            
            try:
                db.save_simulation(request.prompt, sim_data)
            except Exception as db_err:
                print(f"Database logging failed: {db_err}")

            simulations.append(sim_data)
        
        return {
            "status": "success",
            "model_tested": "gemini-1.5-flash-latest",
            "simulations": simulations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")