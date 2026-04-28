#!/usr/bin/env python
"""
Entry point to run the ML Project API server.

Usage:
    python main.py
    # or for development with auto-reload:
    uvicorn main:app --reload
"""

# Load .env BEFORE importing anything else
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from src.server import app

if __name__ == "__main__":
    # Run the server on localhost:8000
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )

