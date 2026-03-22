from fastapi import FastAPI
from app.routers import monitor

app = FastAPI(title="SimuRed API")

app.include_router(monitor.router)

@app.get("/")
async def root():
    return {"message" : "SimuRed System Active"}