#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Dashboard - Main Entry Point
A comprehensive portfolio management and risk analysis platform
"""

import os
import sys
import uvicorn
from pathlib import Path

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "portfolio_management"))
sys.path.append(str(project_root / "shared"))

from portfolio_management.portfolio_api import app

if __name__ == "__main__":
    print("🚀 Starting Portfolio Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("🔧 Portfolio Management & Risk Analysis Platform")
    print("=" * 50)
    
    uvicorn.run(
        "portfolio_management.portfolio_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
