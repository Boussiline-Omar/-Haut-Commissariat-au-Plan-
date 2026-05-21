# Guide d'Intégration Backend ↔ Frontend

## 📋 Vue d'ensemble

Vous avez maintenant un système complet **Backend FastAPI** + **Frontend HTML/JavaScript** connectés via des appels API.

### Architecture
```
Frontend (rgph_dashboard.html)
    ↓ fetch() API calls
Backend (FastAPI - main.py)
    ↓ JSON responses
Frontend (affiche les données)
```

## 🚀 Démarrage rapide

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Démarrer le serveur
```bash
python run_server.py 8000
```

Ou directement avec uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Accéder au dashboard
- **Dashboard**: http://localhost:8000/
- **Documentation API**: http://localhost:8000/docs
- **Redoc**: http://localhost:8000/redoc

---

## 🔌 Points de connexion

### Frontend → Backend

#### 1. **Chargement des régions** (au démarrage)
```javascript
GET /segmentation/regions
```
- Charge les données de toutes les régions
- Récupère: cluster, taux_emploi, niv_edu, score_continu, etc.
- Appelée dans `loadRegionData()`

#### 2. **Chargement des KPIs** (au démarrage)
```javascript
GET /stats/dashboard
```
- Récupère les indicateurs clés nationaux
- Affiche en haut du dashboard
- Appelée dans `loadDashboardKPI()`

#### 3. **Scoring d'individu** (simulateur)
```javascript
POST /predict/individu
Body: {
  reg: number,
  milieu: number,
  NIV_ETU_AGR: number,
  TY_ACT: number,
  SIT_HANDICAP: number,
  LIR_ECR: number
}
Response: {
  classe_vuln_indiv: 0|1|2,
  score_continu: float,
  proba_0: float,
  proba_1: float,
  proba_2: float
}
```
- Appelée dans `runSim()` quand l'utilisateur clique "Calculer le score"

---

## 📊 Flot de données

### Étape 1: Chargement initial
```
Page charge
  ↓
window.addEventListener('load', initDashboard)
  ↓
loadRegionData() ─→ GET /segmentation/regions
loadDashboardKPI() ─→ GET /stats/dashboard
  ↓
buildBarList() + buildFeatGrid()
Graphiques Chart.js générés avec les vraies données
```

### Étape 2: Interaction utilisateur
```
Utilisateur remplit le formulaire du simulateur
  ↓
Clique "Calculer le score"
  ↓
runSim() → POST /predict/individu
  ↓
Reçoit les prédictions du backend
  ↓
Affiche le résultat avec probabilités
```

---

## 🛠️ Fichiers modifiés

### Backend (`main.py`)
✅ **Ajouts:**
- Import: `StaticFiles`, `FileResponse`
- Route `GET /`: Serve le fichier `rgph_dashboard.html`
- CORS déjà configuré pour accepter les requêtes depuis le frontend

### Frontend (`rgph_dashboard.html`)
✅ **Modifications:**
- Variables `REGIONS` et `DASHBOARD_KPI` déclarées dynamiquement
- Nouvelles fonctions:
  - `loadRegionData()`: Récupère `/segmentation/regions`
  - `loadDashboardKPI()`: Récupère `/stats/dashboard`
  - `updateKPIDisplay()`: Met à jour l'affichage des KPIs
  - `initDashboard()`: Initialise au chargement
- `runSim()` modifié pour appeler `/predict/individu` au lieu de calculer localement

---

## ✅ Checklist de test

- [ ] Backend démarre sans erreur: `python run_server.py`
- [ ] Accédez http://localhost:8000 → dashboard s'affiche
- [ ] Console JS: Pas d'erreur CORS
- [ ] Les régions se chargent (`console.log` affiche les données)
- [ ] Les KPIs s'affichent en haut
- [ ] Le graphique de bars montre les top régions
- [ ] Le simulateur: Remplissez et cliquez "Calculer"
- [ ] Le résultat affiche les prédictions du backend

---

## 🔍 Debugging

### Erreur CORS
```
Access to XMLHttpRequest at 'http://localhost:8000/segmentation/regions' 
from origin 'http://localhost:8000' has been blocked by CORS policy
```
**Solution**: CORS est déjà configuré dans `main.py`. Vérifiez que le backend redémarre.

### Erreur 404
```
POST http://localhost:8000/predict/individu 404
```
**Solution**: Vérifiez que le endpoint existe dans `main.py` (regardez les décorateurs `@app.post`)

### Données vides
**Solution**: 
1. Ouvrez les DevTools (F12)
2. Onglet "Network" → Vérifiez les requêtes `segmentation/regions` et `stats/dashboard`
3. Vérifiez que le backend retourne du JSON valide

---

## 📝 Notes importantes

1. **Données dynamiques**: Toutes les données proviennent du backend maintenant
2. **CORS configuré**: Accepte toutes les origines (À restreindre en production!)
3. **Modèles ML**: S'assurez que les fichiers dans `outputs/` existent pour les chargements
4. **Production**: Déployez avec Gunicorn + Nginx (pas uvicorn avec reload=True)

---

## 🎯 Prochaines étapes possibles

1. ✅ **Authentification**: Ajouter JWT pour sécuriser les endpoints
2. ✅ **Base de données**: Stocker les prédictions dans PostgreSQL
3. ✅ **Cache**: Mettre en cache les régions/KPIs avec Redis
4. ✅ **Frontend React**: Migrer vers une SPA React + API
5. ✅ **Monitoring**: Ajouter des logs pour tracer les erreurs

---

Bonne chance! 🚀
