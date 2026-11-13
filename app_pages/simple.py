"""Décision simple : saisir les critères et la matrice de décision, obtenir le classement."""
import numpy as np
import pandas as pd
import streamlit as st

from interface.commun import Donnees, choix_poids, editeur_persistant, remplacer_editeur
from mcdm.classement import METHODES

SENS = {"Bénéfice (+)": 1, "Coût (−)": -1}

CRITERES_DEPART = pd.DataFrame(
    {
        "Critère": ["Style", "Fiabilité", "Économie de carburant", "Coût"],
        "Sens": ["Bénéfice (+)", "Bénéfice (+)", "Bénéfice (+)", "Coût (−)"],
        "Poids": [0.2, 0.1, 0.4, 0.3],
    }
)
MATRICE_DEPART = pd.DataFrame(
    {
        "Alternative": ["M1", "M2", "M3", "M4"],
        "Style": [7.0, 8, 9, 6],
        "Fiabilité": [9.0, 7, 6, 7],
        "Économie de carburant": [9.0, 8, 8, 8],
        "Coût": [8.0, 7, 9, 6],
    }
)


def arreter(message):
    st.error(message, icon=":material/error:")
    st.stop()


st.caption("Saisissez vos critères et votre matrice de décision : le classement s'affiche en dessous.")

# --- 1. Critères ----------------------------------------------------------------
st.subheader("1. Critères")
criteres = editeur_persistant(
    "simple_criteres",
    CRITERES_DEPART,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Critère": st.column_config.TextColumn("Critère", required=True),
        "Sens": st.column_config.SelectboxColumn("Sens", options=list(SENS), required=True, default="Bénéfice (+)"),
        "Poids": st.column_config.NumberColumn("Poids", min_value=0.0, format="%.3f", required=True, default=1.0),
    },
)
criteres = criteres[criteres["Critère"].fillna("").astype(str).str.strip() != ""]
noms = [str(c).strip() for c in criteres["Critère"]]
if len(noms) < 2:
    arreter("Ajoutez au moins deux critères.")
if len(set(noms)) != len(noms):
    arreter("Deux critères portent le même nom.")

# --- 2. Matrice de décision -------------------------------------------------------
st.subheader("2. Matrice de décision")
st.caption("Une ligne par alternative, une colonne par critère. Ajoutez des lignes en bas du tableau.")

# Les colonnes de la matrice suivent la liste des critères ; les valeurs déjà saisies sont gardées.
signature = "|".join(noms)
if st.session_state.get("simple_matrice_signature", signature) != signature:
    precedente = st.session_state["simple_matrice__valeur"]
    anciennes = [c for c in precedente.columns if c != "Alternative"]
    if len(anciennes) == len(noms):  # même nombre de critères : c'est un renommage
        precedente = precedente.rename(columns=dict(zip(anciennes, noms)))
    remplacer_editeur("simple_matrice", precedente.reindex(columns=["Alternative", *noms]))
st.session_state["simple_matrice_signature"] = signature

matrice = editeur_persistant(
    "simple_matrice",
    MATRICE_DEPART,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Alternative": st.column_config.TextColumn("Alternative", required=True),
        **{c: st.column_config.NumberColumn(c) for c in noms},
    },
)

matrice = matrice[matrice["Alternative"].fillna("").astype(str).str.strip() != ""]
alternatives = [str(a).strip() for a in matrice["Alternative"]]
if len(alternatives) < 2:
    arreter("Ajoutez au moins deux alternatives.")
if len(set(alternatives)) != len(alternatives):
    arreter("Deux alternatives portent le même nom.")
X = matrice[noms].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
if not np.isfinite(X).all():
    arreter("Remplissez toutes les cases de la matrice avec des nombres.")

sens = np.array([SENS[s] for s in criteres["Sens"]])
poids = criteres["Poids"].to_numpy(dtype=float)
if not np.isfinite(poids).all() or (poids < 0).any() or poids.sum() <= 0:
    arreter("Les poids doivent être positifs, avec une somme supérieure à 0.")
poids = poids / poids.sum()

# --- 3. Poids ---------------------------------------------------------------------
st.subheader("3. Poids des critères")
st.caption("Gardez les poids du tableau des critères, ou calculez-les avec une méthode de pondération.")
poids = choix_poids("simple", Donnees(alternatives, noms, sens, X, poids))

# --- 4. Résultat ------------------------------------------------------------------
st.subheader("4. Résultat")
methode = st.segmented_control("Méthode", list(METHODES), default="TOPSIS", key="simple_methode") or "TOPSIS"

try:
    res = METHODES[methode](X, poids, sens)
except ValueError as exc:
    arreter(f"{methode} ne s'applique pas à ces données : {exc}")

gagnants = [a for a, r in zip(alternatives, res["rangs"]) if r == 1]
st.success(f"Meilleur choix : **{' et '.join(gagnants)}**", icon=":material/emoji_events:")

titre_score = "Q (plus petit = meilleur)" if methode == "VIKOR" else "Score"
resultat = (
    pd.DataFrame({"Rang": res["rangs"].astype(int), "Alternative": alternatives, titre_score: res["scores"]})
    .sort_values(["Rang", "Alternative"])
)
st.dataframe(
    resultat,
    hide_index=True,
    column_config={titre_score: st.column_config.NumberColumn(format="%.4f")},
)
