"""Tableau de bord Streamlit : interroge l'API de prévision et présente les performances du modèle.

    streamlit run src/conso/dashboard/app.py

Variables d'environnement :
- ``CONSO_API_URL``  : adresse de l'API (défaut : http://localhost:8000) ;
- ``CONSO_DATASET``  : jeu de données (page « Performance ») ;
- ``CONSO_BACKTEST`` : prédictions du backtest de production (page « Performance »).
"""
