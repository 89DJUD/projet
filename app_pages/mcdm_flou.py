import numpy as np
import pandas as pd
import streamlit as st

from interface import graphiques as gr
from interface.commun import SENS_LIBELLES, SENS_VALEURS, executer, resultat_principal, stop_si_erreur, tableau_nombres
from mcdm import exemples as ex
from mcdm import mcdm_flou as mf

vue = st.segmented_control(
    "Méthode", ["Score flou d'une alternative", "TOPSIS flou", "TOPSIS intuitionniste"], default="TOPSIS flou", key="mf_vue"
) or "TOPSIS flou"


def triplets(tableau):
    """Affiche un tableau de triplets flous sous forme lisible."""
    return np.vectorize(lambda a, b, c: f"({a:.3f} ; {b:.3f} ; {c:.3f})")(tableau[..., 0], tableau[..., 1], tableau[..., 2])


def couples(tableau):
    return np.vectorize(lambda m, n: f"({m:.3f} ; {n:.3f})")(tableau[..., 0], tableau[..., 1])


def saisie_linguistique():
    """Tableaux partagés par les deux TOPSIS : notes de chaque décideur et poids des critères."""
    with st.expander("Échelles linguistiques"):
        gauche, droite = st.columns(2)
        gauche.dataframe(
            pd.DataFrame(
                [{"Note": k, "Libellé": mf.LIBELLES_NOTES[k], "Nombre flou": str(v)} for k, v in mf.ECHELLE_NOTES.items()]
            ),
            hide_index=True,
        )
        droite.dataframe(
            pd.DataFrame(
                [{"Poids": k, "Libellé": mf.LIBELLES_POIDS[k], "Nombre flou": str(v)} for k, v in mf.ECHELLE_POIDS.items()]
            ),
            hide_index=True,
        )
    criteres = list(ex.RECRUTEMENT_CRITERES)
    st.markdown("**Sens et importance de chaque critère selon chaque décideur**")
    table_poids = pd.DataFrame(np.array(ex.RECRUTEMENT_POIDS).T, columns=list(ex.RECRUTEMENT_DECIDEURS))
    table_poids.insert(0, "Critère", criteres)
    table_poids.insert(1, "Sens", SENS_LIBELLES[1])
    config_poids = {
        dm: st.column_config.SelectboxColumn(dm, options=list(mf.ECHELLE_POIDS), required=True)
        for dm in ex.RECRUTEMENT_DECIDEURS
    }
    config_poids["Sens"] = st.column_config.SelectboxColumn("Sens", options=list(SENS_VALEURS), required=True)
    table_poids = st.data_editor(table_poids, hide_index=True, disabled=["Critère"], column_config=config_poids, key="mf_poids")
    if table_poids.isna().any().any():
        stop_si_erreur("Complétez le sens et l'importance de chaque critère.")

    st.markdown("**Évaluation des candidats par chaque décideur**")
    onglets = st.tabs(list(ex.RECRUTEMENT_DECIDEURS))
    notes = []
    config_notes = {
        c: st.column_config.SelectboxColumn(c, options=list(mf.ECHELLE_NOTES), required=True) for c in criteres
    }
    for k, onglet in enumerate(onglets):
        with onglet:
            lignes = pd.DataFrame(
                [ex.RECRUTEMENT_NOTES[i][k] for i in range(len(ex.RECRUTEMENT_ALTERNATIVES))],
                columns=criteres,
                index=list(ex.RECRUTEMENT_ALTERNATIVES),
            )
            notes.append(st.data_editor(lignes, column_config=config_notes, key=f"mf_notes_{k}"))
    if any(n.isna().any().any() for n in notes):
        stop_si_erreur("Complétez toutes les évaluations.")
    notes_ling = [
        [notes[k].iloc[i].tolist() for k in range(len(notes))] for i in range(len(ex.RECRUTEMENT_ALTERNATIVES))
    ]
    poids_ling = [table_poids[dm].tolist() for dm in ex.RECRUTEMENT_DECIDEURS]
    sens = table_poids["Sens"].map(SENS_VALEURS).to_numpy()
    return criteres, list(ex.RECRUTEMENT_ALTERNATIVES), notes_ling, poids_ling, sens


if vue == "Score flou d'une alternative":
    st.caption("Évaluer une seule alternative avec des nombres triangulaires (a pessimiste, b probable, c optimiste).")
    table = st.data_editor(
        pd.DataFrame(
            {
                "Critère": ["ROA", "Liquidité"],
                "Poids": [0.6, 0.4],
                "a": [1.0, 110.0],
                "b": [1.2, 120.0],
                "c": [1.5, 130.0],
                "c max": [1.5, 130.0],
            }
        ),
        hide_index=True,
        num_rows="dynamic",
        column_config={"c max": st.column_config.NumberColumn("c max", help="Plus grand c parmi les alternatives.")},
        key="mf_score",
    ).dropna()
    if table.empty:
        stop_si_erreur("Indiquez au moins un critère complet.")
    if not np.isclose(table["Poids"].sum(), 1, atol=1e-3):
        stop_si_erreur(f"La somme des poids doit valoir 1 (elle vaut {table['Poids'].sum():.3f}).")
    res = executer(mf.score_flou, table[["a", "b", "c"]].to_numpy(dtype=float), table["Poids"].to_numpy(dtype=float), table["c max"].to_numpy(dtype=float))
    a, b, c = st.columns(3)
    a.metric("Score agrégé S̃", f"({res['agrege'][0]:.2f} ; {res['agrege'][1]:.2f} ; {res['agrege'][2]:.2f})")
    b.metric("Score défuzzifié S", f"{res['score']:.3f}", help="Centre de gravité (a + b + c) / 3.")
    c.metric("Incertitude c − a", f"{res['incertitude']:.3f}", help="À surveiller en plus du score.")
    st.markdown("Matrice normalisée :")
    tableau_nombres(pd.DataFrame(res["normalisee"], index=table["Critère"], columns=["a", "b", "c"]), decimales=3)

elif vue == "TOPSIS flou":
    st.caption("Chen, 2000. Exemple : recrutement d'un ingénieur par trois décideurs.")
    criteres, alternatives, notes_ling, poids_ling, sens = saisie_linguistique()
    notes = mf.notes_vers_tableau(notes_ling, mf.ECHELLE_NOTES)
    poids = mf.poids_vers_tableau(poids_ling, mf.ECHELLE_POIDS)
    res = executer(mf.topsis_flou, notes, poids, sens)
    resultat_principal(alternatives, res["rangs"], f"Coefficient de rapprochement : {res['scores'].max():.4f}.")
    gr.barres(alternatives, res["scores"], "Coefficient CC")
    tableau_nombres(
        pd.DataFrame({"D*": res["d_plus"], "D⁻": res["d_moins"], "CC": res["scores"], "Rang": res["rangs"]}, index=alternatives).sort_values("Rang")
    )
    with st.expander("Détail des calculs"):
        st.markdown("**1. Matrice de décision collective**")
        st.dataframe(pd.DataFrame(triplets(res["collective"]), index=alternatives, columns=criteres))
        st.markdown("**2. Matrice collective normalisée**")
        st.dataframe(pd.DataFrame(triplets(res["normalisee"]), index=alternatives, columns=criteres))
        st.markdown("**3. Poids collectifs**")
        st.dataframe(pd.DataFrame(triplets(res["poids"][None, :, :]), index=["Poids"], columns=criteres))
        st.markdown("**4. Matrice normalisée pondérée**")
        st.dataframe(pd.DataFrame(triplets(res["ponderee"]), index=alternatives, columns=criteres))
        st.markdown("**5. Résolution** : idéal positif (1, 1, 1), idéal négatif (0, 0, 0)")
        st.latex(r"d(v,v^*)=\sqrt{\tfrac13\sum_{k=1}^3\left(v^k-1\right)^2},\quad CC_i=\frac{D_i^-}{D_i^-+D_i^*}")

else:
    st.caption("Échelle intuitionniste choisie pour cet exemple, appliquée au même recrutement que TOPSIS flou.")
    criteres, alternatives, notes_ling, poids_ling, sens = saisie_linguistique()
    with st.expander("Échelle intuitionniste utilisée"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Note": k, "Poids": p, "μ": v[0], "ν": v[1], "π": round(1 - v[0] - v[1], 2)}
                    for (k, v), p in zip(mf.ECHELLE_NOTES_IF.items(), mf.ECHELLE_POIDS_IF)
                ]
            ),
            hide_index=True,
        )
    st.markdown("**Poids des décideurs σ**")
    colonnes = st.columns(len(ex.RECRUTEMENT_DECIDEURS))
    sigma = np.array(
        [c.number_input(dm, 0.0, 1.0, round(1 / 3, 3), 0.01, key=f"mf_sigma_{dm}") for c, dm in zip(colonnes, ex.RECRUTEMENT_DECIDEURS)]
    )
    if sigma.sum() <= 0:
        stop_si_erreur("La somme des poids des décideurs doit être strictement positive.")
    sigma = sigma / sigma.sum()
    notes = mf.notes_vers_tableau(notes_ling, mf.ECHELLE_NOTES_IF)
    poids = mf.poids_vers_tableau(poids_ling, mf.ECHELLE_POIDS_IF)
    res = executer(mf.topsis_intuitionniste, notes, poids, sens, sigma)
    resultat_principal(alternatives, res["rangs"], f"Coefficient de proximité : {res['scores'].max():.4f}.")
    gr.barres(alternatives, res["scores"], "Coefficient de proximité C")
    tableau_nombres(
        pd.DataFrame({"S⁺": res["s_plus"], "S⁻": res["s_moins"], "C": res["scores"], "Rang": res["rangs"]}, index=alternatives).sort_values("Rang")
    )
    with st.expander("Détail des calculs"):
        st.latex(r"\alpha_{ij}=\left(1-\prod_k(1-\mu_{ij}^k)^{\sigma_k},\ \prod_k(\nu_{ij}^k)^{\sigma_k}\right)")
        st.markdown("Matrice agrégée par IFWA :")
        st.dataframe(pd.DataFrame(couples(res["agregee"]), index=alternatives, columns=criteres))
        st.markdown("Poids agrégés :")
        st.dataframe(pd.DataFrame(couples(res["poids"][None, :, :]), index=["Poids"], columns=criteres))
        st.markdown("Matrice agrégée pondérée :")
        st.dataframe(pd.DataFrame(couples(res["ponderee"]), index=alternatives, columns=criteres))
        st.markdown("Solutions idéales :")
        st.dataframe(pd.DataFrame(np.array([couples(res["ideal"]), couples(res["anti_ideal"])]), index=["I⁺", "I⁻"], columns=criteres))
        st.latex(r"S_i^{\pm}=\frac{1}{2m}\sum_j\sqrt{(\mu_{ij}-\mu_j^{\pm})^2+(\nu_{ij}-\nu_j^{\pm})^2+(\pi_{ij}-\pi_j^{\pm})^2}")
