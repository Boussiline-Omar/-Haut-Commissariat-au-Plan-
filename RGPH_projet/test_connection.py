#!/usr/bin/env python
"""
Test script pour vérifier la connexion Backend ↔ Frontend
Exécutez ce script APRÈS avoir démarré le serveur:
    python test_connection.py
"""
import requests
import json
import time
from typing import Optional

BASE_URL = "http://localhost:8000"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{bcolors.BOLD}{bcolors.HEADER}{'='*60}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{text}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}{'='*60}{bcolors.ENDC}\n")

def print_success(text):
    print(f"{bcolors.OKGREEN}✅ {text}{bcolors.ENDC}")

def print_error(text):
    print(f"{bcolors.FAIL}❌ {text}{bcolors.ENDC}")

def print_info(text):
    print(f"{bcolors.OKCYAN}ℹ️  {text}{bcolors.ENDC}")

def test_endpoint(method: str, endpoint: str, data: Optional[dict] = None, show_response: bool = False):
    """Test un endpoint de l'API"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        else:
            print_error(f"Méthode HTTP inconnue: {method}")
            return False

        if response.status_code == 200:
            print_success(f"{method} {endpoint} → 200 OK")
            if show_response:
                try:
                    print_info(f"Réponse: {json.dumps(response.json(), indent=2)[:200]}...")
                except:
                    print_info(f"Réponse: {response.text[:200]}...")
            return True
        else:
            print_error(f"{method} {endpoint} → {response.status_code}")
            print_info(f"Message: {response.text[:100]}")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"❌ Connexion refusée. Le serveur démarre-t-il sur {BASE_URL}?")
        return False
    except Exception as e:
        print_error(f"{method} {endpoint} → Erreur: {str(e)}")
        return False

def main():
    print_header("🧪 TEST DE CONNEXION BACKEND ↔ FRONTEND")
    
    print_info(f"Base URL: {BASE_URL}")
    print_info("Vérification des connexions...\n")
    
    # Test 1: Santé du serveur
    print_header("1️⃣ Santé du serveur")
    health = test_endpoint("GET", "/", show_response=False)
    
    if not health:
        print_error("Le serveur ne répond pas. Vérifiez qu'il a démarré avec 'python run_server.py'")
        return
    
    time.sleep(0.5)
    
    # Test 2: Chargement des régions
    print_header("2️⃣ Chargement des régions")
    regions = test_endpoint("GET", "/segmentation/regions", show_response=True)
    
    time.sleep(0.5)
    
    # Test 3: Chargement des KPIs
    print_header("3️⃣ Chargement des KPIs")
    kpis = test_endpoint("GET", "/stats/dashboard", show_response=True)
    
    time.sleep(0.5)
    
    # Test 4: Profil région
    print_header("4️⃣ Profil d'une région")
    region_profile = test_endpoint("GET", "/stats/region/6", show_response=True)
    
    time.sleep(0.5)
    
    # Test 5: Prédiction individu
    print_header("5️⃣ Prédiction individu (scoring)")
    test_data = {
        "reg": 6,
        "milieu": 1,
        "NIV_ETU_AGR": 3,
        "TY_ACT": 1,
        "SIT_HANDICAP": 1,
        "LIR_ECR": 1
    }
    print_info(f"Données envoyées: {json.dumps(test_data, indent=2)}")
    prediction = test_endpoint("POST", "/predict/individu", data=test_data, show_response=True)
    
    time.sleep(0.5)
    
    # Test 6: Prédiction ménage
    print_header("6️⃣ Prédiction ménage")
    menage_data = {
        "reg": 6,
        "milieu": 1,
        "typ_menage": 1,
        "nbre_membres": 4,
        "pct_enfants": 0.25,
        "pct_inactifs": 0.3
    }
    print_info(f"Données envoyées: {json.dumps(menage_data, indent=2)}")
    menage_pred = test_endpoint("POST", "/predict/menage", data=menage_data, show_response=True)
    
    # Résumé
    print_header("📊 RÉSUMÉ DES TESTS")
    
    results = {
        "Santé serveur": health,
        "Régions": regions,
        "KPIs": kpis,
        "Profil région": region_profile,
        "Scoring individu": prediction,
        "Scoring ménage": menage_pred,
    }
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n{bcolors.BOLD}Résultats:{bcolors.ENDC}")
    for test_name, result in results.items():
        status = f"{bcolors.OKGREEN}✅ PASS{bcolors.ENDC}" if result else f"{bcolors.FAIL}❌ FAIL{bcolors.ENDC}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{bcolors.BOLD}Total: {passed}/{total} tests réussis{bcolors.ENDC}")
    
    if passed == total:
        print(f"\n{bcolors.OKGREEN}{bcolors.BOLD}🎉 EXCELLENT! Tous les tests sont passés!{bcolors.ENDC}")
        print(f"{bcolors.OKCYAN}Vous pouvez maintenant accéder au dashboard: {BASE_URL}/{bcolors.ENDC}\n")
    else:
        print(f"\n{bcolors.WARNING}{bcolors.BOLD}⚠️  Certains tests ont échoué.{bcolors.ENDC}")
        print(f"{bcolors.OKCYAN}Vérifiez les logs du serveur pour plus de détails.\n{bcolors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{bcolors.WARNING}Tests interrompus par l'utilisateur.{bcolors.ENDC}\n")
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
