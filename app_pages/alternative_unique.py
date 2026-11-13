import pandas as pd
import streamlit as st

from interface import graphiques as gr
from interface.commun import SENS_LIBELLES, SENS_VALEURS, executer, normaliser_poids, stop_si_erreur, tableau_nombres
from mcdm import alternative_unique as au
from mcdm import exemples as ex

vue = st.segmented_control(
    "Technique", ["Score MAUT et décision", "Choisir le seuil (ROC)"], default="Score MAUT et décision", key="uni_vue"
) or "Score MAUT et décision"

if vue == "Score MAUT et décision":
    st.caption("Exemple d'illustration : scoring d'un dossier de crédit.")
    table = st.data_editor(
        pd.DataFrame(
            {
                "Critère": ["Revenu mensuel (DH)", "Taux d'endettement (%)", "Ancienneté (années)"],
                "Sens": [SENS_LIBELLES[1], SENS_LIBELLES[-1], SENS_LIBELLES[1]],
                "Poids": [0.5, 0.3, 0.2],
                "Valeur": [12000.0, 28.0, 6.0],
                "Borne min": [3000.0, 0.0, 0.0],
                "Borne max": [30000.0, 60.0, 20.0],
            }
        ),
        hide_index=True,
        num_rows="dynamic",
        column_config={"Sens": st.column_config.SelectboxColumn("Sens", options=list(SENS_VALEURS), required=True)},
        key="uni_table",
    ).dropna()
    if len(table) < 1:
        stop_si_erreur("Indiquez au moins un critère complet.")
    w = normaliser_poids(table["Poids"].to_numpy(dtype=float))
    sens = table["Sens"].map(SENS_VALEURS).to_numpy()
    args = (table["Valeur"].to_numpy(dtype=float), w, sens, table["Borne min"].to_numpy(dtype=float), table["Borne max"].to_numpy(dtype=float))
    res = executer(au.maut, *args)
    tau = st.slider("Seuil de décision τ", 0.0, 1.0, 0.7, 0.01, key="uni_tau")
    if res["score"] >= tau:
        st.success(f"**Accepter** : S(A) = {res['score']:.3f} ≥ τ = {tau:.2f}.", icon=":material/check_circle:")
    else:
        st.warning(f"**Rejeter** : S(A) = {res['score']:.3f} < τ = {tau:.2f}.", icon=":material/cancel:")
    gr.barres(table["Critère"], res["utilites"], "Utilité uj(xj), entre 0 et 1")
    st.latex(r"S(A)=\sum_j w_j\,u_j(x_j),\quad u_j=\frac{x-x_j^{min}}{x_j^{max}-x_j^{min}}\ (+),\quad u_j=\frac{x-x_j^{max}}{x_j^{min}-x_j^{max}}\ (-)")
    ideal = executer(au.distance_ideal, *args)
    st.markdown("**Distance à l'idéal, logique TOPSIS**")
    a, b, c = st.columns(3)
    a.metric("D⁺ (distance à l'idéal)", f"{ideal['d_plus']:.3f}")
    b.metric("D⁻ (distance à l'anti-idéal)", f"{ideal['d_moins']:.3f}")
    c.metric("Indice de performance C", f"{ideal['indice']:.3f}")
else:
    st.caption("Exemple : six décisions passées, à compléter avec vos propres données.")
    table = st.data_editor(
        pd.DataFrame(
            {
                "Alternative": ex.SEUIL_ALTERNATIVES,
                "Score S": ex.SEUIL_SCORES,
                "Accepter": [bool(v) for v in ex.SEUIL_DECISIONS],
            }
        ),
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Score S": st.column_config.NumberColumn("Score S", min_value=0.0, max_value=1.0, format="%.2f"),
            "Accepter": st.column_config.CheckboxColumn("Décision passée : accepter", default=False),
        },
        key="seuil_table",
    ).dropna(subset=["Score S"])
    col_fp, col_fn = st.columns(2)
    cout_fp = col_fp.number_input("Coût d'un faux positif", 0.0, value=1.0, step=0.5, key="seuil_fp", help="Accepter à tort.")
    cout_fn = col_fn.number_input("Coût d'un faux négatif", 0.0, value=1.0, step=0.5, key="seuil_fn", help="Rejeter à tort.")
    scores = table["Score S"].to_numpy(dtype=float)
    decisions = table["Accepter"].fillna(False).astype(int).to_numpy()
    lignes = executer(au.table_roc, scores, decisions, cout_fp, cout_fn)
    meilleurs = au.meilleurs_seuils(lignes)
    critere = st.pills("Critère de choix du seuil", list(meilleurs), default="Indice de Youden", key="seuil_critere") or "Indice de Youden"
    retenu = meilleurs[critere]
    st.success(
        f"**Seuil retenu : τ = {retenu['seuil']:.3f}.** TPR = {retenu['TPR']:.2f}, FPR = {retenu['FPR']:.2f}, "
        f"exactitude = {retenu['exactitude']:.0%}.",
        icon=":material/flag:",
    )
    gr.courbe_roc(lignes, retenu)
    st.caption(f"Aire sous la courbe (AUC) : {au.auc(lignes):.3f}.")
    st.markdown("**Seuils testés**")
    colonnes = ["seuil", "TP", "FP", "FN", "TN", "TPR", "FPR", "distance", "youden", "exactitude", "cout"]
    tableau_nombres(pd.DataFrame(lignes)[colonnes], decimales=3, hide_index=True)
