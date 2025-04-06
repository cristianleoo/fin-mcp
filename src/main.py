from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from src.api.routes import router

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Financial Analysis MCP Server",
    description="A server for financial analysis capabilities including stock data retrieval and visualization",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Financial Analysis MCP Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 