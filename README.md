# Application Streamlit : prévision du rendement mensuel du S&P 500 (LSTM)

Fichiers du dossier :
- `app.py` : l'application
- `requirements.txt` : les bibliothèques à installer
- `modele_lstm_final.keras` et `config_finale.json` : le modèle entraîné (créés par le notebook des étapes 9 et 10)
- `scaler.json` : moyenne et écart-type de la normalisation (calculés sur la période d'entraînement)
- `sp500_multivarie_propre_1881_2023.csv` : les données (Robert Shiller, https://datahub.io/core/s-and-p-500)

Lancer l'application sur votre ordinateur :
    pip install -r requirements.txt
    streamlit run app.py
