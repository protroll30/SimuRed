import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class DatabaseService:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)

    def save_simulation(self, original: str, result_dict: dict):
        """
        Pushes a single simulation result to the Supabase 'simulations' table.
        """
        data = {
            "original_prompt": original,
            "attack_type": result_dict["type"],
            "mutated_input": result_dict["input"],
            "ai_output": result_dict["output"],
            "similarity_score": result_dict["drift_metrics"].get("similarity_score", 1.0),
            "is_stable": result_dict["drift_metrics"].get("is_stable", True)
        }
        
        # Insert the dictionary into the table
        return self.supabase.table("simulations").insert(data).execute()