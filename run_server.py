#!/usr/bin/env python
"""
Script de démarrage du serveur FastAPI avec le dashboard.
Usage: python run_server.py [port]
"""
import uvicorn
import sys

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    print(f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║         RGPH 2014 — Serveur Frontend + Backend FastAPI         ║
    ╚════════════════════════════════════════════════════════════════╝
    
    🌐 Dashboard   : http://localhost:{port}/
    📚 API Docs    : http://localhost:{port}/docs
    🔄 API ReDoc   : http://localhost:{port}/redoc
    
    Endpoints disponibles:
      GET  /                          → Dashboard HTML
      GET  /segmentation/regions      → Données régions + clusters
      GET  /stats/dashboard           → KPIs globaux
      POST /predict/individu          → Score vulnérabilité
      POST /predict/menage            → Classification ménage
      GET  /stats/region/{{reg_id}}    → Profil région
    
    ✅ Serveur démarrant...
    """)
    
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
