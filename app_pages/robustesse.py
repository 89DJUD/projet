import numpy as np
import pandas as pd
import streamlit as st

from interface import graphiques as gr
from interface.commun import saisie_probleme, stop_si_erreur
from mcdm import classement as cl
from mcdm import exemples as ex
from mcdm import robustesse as ro

FONCTIONS = {"TOPSIS": cl.topsis, "WSM": cl.wsm, "VIKOR": cl.vikor}


vue = st.segmented_control(
    "Analyse", ["Spearman : comparer des classements", "Sensibilité aux poids"], default="Spearman : comparer des classements", key="rob_vue"
) or "Spearman : comparer des classements"

if vue.startswith("Spearman"):
    st.caption("Modifiez les rangs pour comparer vos propres classements.")
    st.latex(r"\rho=1-\frac{6\sum_i d_i^2}{n(n^2-1)}")
    table = st.data_editor(
        pd.DataFrame({"Alternative": ex.SPEARMAN_ALTERNATIVES, **ex.SPEARMAN_RANGS}),
        hide_index=True,
        num_rows="dynamic",
        key="rob_rangs",
    ).dropna()
    colonnes = [c for c in table.columns if c != "Alternative"]
    n = len(table)
    if n < 2:
        stop_si_erreur("Il faut au moins deux alternatives.")
    rangs = {c: pd.to_numeric(table[c], errors="coerce").to_numpy() for c in colonnes}
    if any(np.isnan(v).any() for v in rangs.values()):
        stop_si_erreur("Les rangs doivent être des nombres.")
    noms, M = ro.matrice_spearman(rangs)
    gr.carte_chaleur(pd.DataFrame(M, index=noms, columns=noms), "ρ", domaine=(-1, 1))
    critique = ro.rho_critique(n)
    if critique:
        paires = [
            f"{noms[i]} et {noms[j]} (ρ = {M[i, j]:.2f})"
            for i in range(len(noms))
            for j in range(i + 1, len(noms))
            if M[i, j] >= critique
        ]
        phrase = "Paires significatives : " + ", ".join(paires) + "." if paires else "Aucune paire n'atteint ce seuil."
        st.markdown(f"Avec n = {n}, ρ doit atteindre {critique} pour être significatif à 5 %. {phrase}")
    if n <= 8:
        gr.trajectoires_rangs(table.set_index("Alternative")[colonnes].astype(int))
    consensus = ro.agreger_rangs(rangs)
    premiers = table["Alternative"].to_numpy()[consensus["rang_consensus"] <= 2]
    st.info(
        f"Les rangs moyens placent {', '.join(map(str, premiers))} en tête. Le choix est robuste s'il reste "
        "le même d'une méthode à l'autre ; les rangs qui varient demandent une analyse de sensibilité.",
        icon=":material/lightbulb:",
    )
else:
    d = saisie_probleme("rob", ex.VOITURES_TOPSIS.titre)
    col_m, col_c = st.columns(2)
    nom = col_m.selectbox("Méthode", list(FONCTIONS), key="rob_methode")
    critere = col_c.selectbox("Critère à faire varier", d.criteres, key=f"rob_critere_{'_'.join(d.criteres)}")
    j = d.criteres.index(critere)
    fonction = FONCTIONS[nom]
    try:
        points = ro.balayage_poids(fonction, d.X, d.poids, d.sens, j)
        meilleure, seuils = ro.seuil_inversion(fonction, d.X, d.poids, d.sens)
    except ValueError as exc:
        stop_si_erreur(str(exc))
    if len(d.alternatives) <= 8:
        gr.balayage(
            [p[0] for p in points], [p[1] for p in points], d.alternatives, critere, d.poids[j],
            f"Score {nom}" + (" (petit = mieux)" if nom == "VIKOR" else ""),
        )
    else:
        st.caption("Plus de huit alternatives : la courbe est masquée, les tableaux ci-dessous restent valables.")
    st.markdown(f"**Poids qui ferait perdre la première place à {d.alternatives[meilleure]}**")
    st.dataframe(
        pd.DataFrame(
            {
                "Critère": d.criteres,
                "Poids actuel": np.round(d.poids, 3),
                "Poids d'inversion le plus proche": [f"{s:.2f}" if s is not None else "Aucun" for s in seuils],
            }
        ),
        hide_index=True,
    )
