import json

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

st.set_page_config(page_title="Prévision du S&P 500 (LSTM)", page_icon="📈", layout="wide")

FICHIER_DONNEES = "sp500_multivarie_propre_1881_2023.csv"
COLONNES = ["SP500", "Earnings", "Consumer Price Index", "Long Interest Rate", "PE10"]
NB_MOIS_TABLE = 13   # 13 mois de niveaux donnent 12 mois de variations


@st.cache_resource
def charger_modele():
    modele = keras.models.load_model("modele_lstm_final.keras")
    with open("scaler.json") as fichier:
        scaler = json.load(fichier)          # moyenne et écart-type calculés sur l'entraînement
    with open("config_finale.json") as fichier:
        cfg = json.load(fichier)
    return modele, scaler, cfg


@st.cache_data
def charger_donnees():
    df = pd.read_csv(FICHIER_DONNEES, parse_dates=["Date"])
    return df.set_index("Date").sort_index()[COLONNES]


def preparer_entree(table, scaler, fenetre):
    """Transforme les niveaux mensuels en variations, normalise et crée l'entrée du LSTM."""
    t = table.astype(float)
    caracteristiques = pd.DataFrame({
        "ret":      np.log(t["SP500"]).diff(),
        "earn_g":   np.log(t["Earnings"]).diff(),
        "infl":     np.log(t["Consumer Price Index"]).diff(),
        "rate_d":   t["Long Interest Rate"].diff(),
        "pe10_log": np.log(t["PE10"]),
    }).dropna()
    moyenne, ecart = np.array(scaler["mean"]), np.array(scaler["scale"])
    X = ((caracteristiques.values - moyenne) / ecart)[-fenetre:]
    return X.reshape(1, fenetre, X.shape[1])


modele, scaler, cfg = charger_modele()
donnees = charger_donnees()
fenetre = int(cfg["fenetre"])

st.title("📈 Prévision du rendement mensuel du S&P 500")
st.caption("Projet Deep Learning : réseau LSTM entraîné sur les données mensuelles de Robert Shiller (1881 à 2023).")

with st.expander("À propos de ce projet"):
    st.markdown(
        f"""
**Problématique.** Prévoir le rendement du mois suivant de l'indice S&P 500 à partir des
{fenetre} derniers mois de cinq variables : rendement de l'indice, croissance des bénéfices,
inflation, variation du taux d'intérêt long et ratio PE10.

**Modèle.** Réseau {cfg['type']} ({cfg['unites']} neurones, dropout {cfg['dropout']}), entraîné avec Keras.

**Données.** Robert Shiller (Université Yale), version distribuée sur datahub.io
(licence ODC-PDDL 1.0).

**Limite importante.** Les rendements boursiers mensuels sont très difficiles à prévoir.
Dans nos tests, ce modèle ne fait pas nettement mieux qu'une prévision par la simple moyenne.
Cette application est un **outil pédagogique** et ne constitue pas un conseil d'investissement.
"""
    )

# ---------- 1. Choix du mois ----------
st.subheader("1. Choisissez le dernier mois connu")
mois_disponibles = [d.strftime("%Y-%m") for d in donnees.index[NB_MOIS_TABLE - 1:]]
mois_choisi = st.select_slider("Dernier mois connu", options=mois_disponibles, value=mois_disponibles[-1])
fin = pd.Timestamp(mois_choisi + "-01")

table_origine = donnees.loc[:fin].tail(NB_MOIS_TABLE).copy()
table_origine.index = table_origine.index.strftime("%Y-%m")

# ---------- 2. Table modifiable ----------
st.subheader("2. Vérifiez ou modifiez les valeurs des 13 derniers mois")
st.write("Vous pouvez modifier les valeurs dans le tableau pour tester d'autres scénarios (toutes les valeurs doivent être positives).")
table = st.data_editor(table_origine, key=f"table_{mois_choisi}")

# ---------- 3. Prévision ----------
st.subheader("3. Prévision pour le mois suivant")

if table.isna().any().any() or (table.astype(float) <= 0).any().any():
    st.error("Toutes les cases doivent être remplies avec des valeurs strictement positives.")
else:
    X = preparer_entree(table, scaler, fenetre)
    prevision = float(modele.predict(X, verbose=0).ravel()[0])      # rendement logarithmique
    dernier_niveau = float(table["SP500"].iloc[-1])
    niveau_prevu = dernier_niveau * float(np.exp(prevision))
    moyenne_historique = float(scaler["mean"][0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Rendement prévu", f"{prevision * 100:+.2f} %")
    c2.metric("Niveau prévu de l'indice", f"{niveau_prevu:,.0f}", f"{niveau_prevu - dernier_niveau:+,.0f} points")
    c3.metric("Rendement moyen (période d'entraînement)", f"{moyenne_historique * 100:+.2f} %")

    if abs(prevision - moyenne_historique) < 0.01:
        st.info("La prévision est proche du rendement moyen historique : le modèle ne détecte pas de signal fort. C'est un comportement attendu pour ce type de données.")

    # Comparaison avec la réalité si le mois suivant existe et si le tableau n'a pas été modifié
    mois_suivant = fin + pd.offsets.MonthBegin(1)
    inchange = np.allclose(table.values.astype(float), table_origine.values.astype(float))
    if mois_suivant in donnees.index and inchange:
        reel = float(np.log(donnees.loc[mois_suivant, "SP500"] / donnees.loc[fin, "SP500"]))
        st.success(
            f"Rendement réel du mois suivant ({mois_suivant.strftime('%Y-%m')}) : {reel * 100:+.2f} %. "
            f"Écart avec la prévision : {(prevision - reel) * 100:+.2f} point(s)."
        )

    st.line_chart(table["SP500"], height=250)
    st.caption("Évolution de l'indice sur les 13 mois utilisés.")
