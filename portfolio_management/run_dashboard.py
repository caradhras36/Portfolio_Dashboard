#!/usr/bin/env python3
"""
Portfolio Dashboard Startup Script
Run this to start the portfolio risk dashboard
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add parent directory to path to import existing modules
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

def main():
    """Start the portfolio dashboard"""
    print("🚀 Starting Portfolio Risk Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "portfolio_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
