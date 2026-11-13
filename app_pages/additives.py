import numpy as np
import pandas as pd
import streamlit as st

from interface import graphiques as gr
from interface.commun import (
    choix_poids,
    executer,
    resultat_principal,
    saisie_paires,
    saisie_probleme,
    table_classement,
    tableau_nombres,
)
from mcdm import classement as cl
from mcdm import exemples as ex
from mcdm import ponderation as po

st.subheader("1. Données", anchor=False)
d = saisie_probleme("add", ex.VOITURES_TOPSIS.titre)

st.subheader("2. Poids des critères", anchor=False)
st.caption("Choisissez comment calculer les poids : à la main, ou avec une méthode de pondération.")
d.poids = choix_poids("add", d)

st.subheader("3. Méthode additive", anchor=False)
methode = st.segmented_control(
    "Méthode", ["WSM, WPM, WASPAS", "TOPSIS", "VIKOR", "AHP complet"], default="TOPSIS", key="add_methode"
) or "TOPSIS"

if methode == "WSM, WPM, WASPAS":
    lam = st.slider("λ : part de WSM dans le score WASPAS", 0.0, 1.0, 0.5, 0.05, key="waspas_lambda")
    res = executer(cl.waspas, d.X, d.poids, d.sens, lam)
    resultat_principal(d.alternatives, res["rangs"], f"Score WASPAS maximal : {res['scores'].max():.4f}.")
    gr.barres(d.alternatives, res["scores"], "Score WASPAS Q")
    tableau_nombres(
        pd.DataFrame(
            {
                "WSM Q(1)": res["q1"],
                "Rang WSM": cl.rangs_depuis_scores(res["q1"]),
                "WPM Q(2)": res["q2"],
                "Rang WPM": cl.rangs_depuis_scores(res["q2"]),
                "WASPAS Q": res["scores"],
                "Rang WASPAS": res["rangs"],
            },
            index=d.alternatives,
        )
    )
    with st.expander("Détail des calculs"):
        st.latex(r"r_{ij}=\frac{x_{ij}}{\max x_j}\ (+),\quad r_{ij}=\frac{\min x_j}{x_{ij}}\ (-)")
        st.latex(r"Q^{(1)}_i=\sum_j w_j r_{ij},\quad Q^{(2)}_i=\prod_j r_{ij}^{w_j},\quad Q_i=\lambda Q^{(1)}_i+(1-\lambda)Q^{(2)}_i")
        tableau_nombres(pd.DataFrame(res["normalisee"], index=d.alternatives, columns=d.criteres))

elif methode == "TOPSIS":
    res = executer(cl.topsis, d.X, d.poids, d.sens)
    resultat_principal(d.alternatives, res["rangs"], f"Coefficient de proximité maximal : {res['scores'].max():.4f}.")
    gr.barres(d.alternatives, res["scores"], "Coefficient de proximité RC*")
    table_classement(d.alternatives, res["scores"], res["rangs"], "RC*")
    with st.expander("Détail des calculs"):
        st.markdown("**Étape 1.** Matrice normalisée")
        st.latex(r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}")
        tableau_nombres(pd.DataFrame(res["normalisee"], index=d.alternatives, columns=d.criteres))
        st.markdown("**Étape 2.** Matrice normalisée et pondérée, $v_{ij}=w_j r_{ij}$")
        tableau_nombres(pd.DataFrame(res["ponderee"], index=d.alternatives, columns=d.criteres))
        st.markdown("**Étape 3.** Solutions idéales")
        tableau_nombres(pd.DataFrame([res["ideal"], res["anti_ideal"]], index=["I⁺", "I⁻"], columns=d.criteres))
        st.markdown("**Étapes 4 et 5.** Séparations et coefficient de proximité")
        st.latex(r"RC^*_i=\frac{S_i^-}{S_i^-+S_i^+}")
        tableau_nombres(
            pd.DataFrame({"S⁺": res["s_plus"], "S⁻": res["s_moins"], "RC*": res["scores"]}, index=d.alternatives)
        )

elif methode == "VIKOR":
    st.caption("Ici, plus Q est petit, meilleure est l'alternative.")
    v = st.slider("v : poids de l'utilité de groupe", 0.0, 1.0, 0.5, 0.05, key="vikor_v")
    res = executer(cl.vikor, d.X, d.poids, d.sens, v)
    compromis = ", ".join(f"**{d.alternatives[i]}**" for i in res["compromis"])
    if len(res["compromis"]) == 1:
        st.success(f"Solution de compromis : {compromis}. Les deux conditions sont respectées.", icon=":material/handshake:")
    else:
        st.warning(f"Ensemble de solutions de compromis : {compromis}.", icon=":material/handshake:")
    a, b = st.columns(2)
    a.metric("Avantage acceptable : Q(A'') − Q(A') ≥ DQ", "Oui" if res["avantage_acceptable"] else "Non", help=f"DQ = 1 / (m − 1) = {res['seuil_dq']:.3f}")
    b.metric("Stabilité acceptable : A' meilleure selon S et R", "Oui" if res["stabilite_acceptable"] else "Non")
    gr.barres(d.alternatives, res["Q"], "Indice Q (petit = mieux)", tri_croissant=True)
    tableau_nombres(
        pd.DataFrame(
            {
                "S (utilité de groupe)": res["S"],
                "Rang S": cl.rangs_depuis_scores(res["S"], False),
                "R (regret individuel)": res["R"],
                "Rang R": cl.rangs_depuis_scores(res["R"], False),
                "Q": res["Q"],
                "Rang Q": res["rangs"],
            },
            index=d.alternatives,
        )
    )
    with st.expander("Détail des calculs"):
        st.latex(r"S_i=\sum_j w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-},\quad R_i=\max_j\left[w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}\right]")
        st.latex(r"Q_i=v\frac{S_i-S^*}{S^--S^*}+(1-v)\frac{R_i-R^*}{R^--R^*}")
        tableau_nombres(pd.DataFrame([res["meilleur"], res["pire"]], index=["x*", "x⁻"], columns=d.criteres))
        st.markdown("Regrets pondérés par critère :")
        tableau_nombres(pd.DataFrame(res["regrets"], index=d.alternatives, columns=d.criteres))

else:
    st.info(
        "AHP complet n'utilise pas les étapes 1 et 2 : il calcule les poids des critères et les priorités "
        "des alternatives à partir de ses propres comparaisons par paires, saisies ci-dessous.",
        icon=":material/info:",
    )
    st.caption("Exemple : achat d'une voiture.")
    col_c, col_a = st.columns(2)
    noms_c = col_c.text_input("Critères", ", ".join(ex.AHP_CRITERES), key="ahpc_criteres")
    noms_a = col_a.text_input("Alternatives", ", ".join(ex.AHP_ALTERNATIVES), key="ahpc_alternatives")
    criteres = [c.strip() for c in noms_c.split(",") if c.strip()]
    alternatives = [a.strip() for a in noms_a.split(",") if a.strip()]
    if len(criteres) < 2 or len(alternatives) < 2 or len(set(criteres)) != len(criteres) or len(set(alternatives)) != len(alternatives):
        st.error("Indiquez au moins deux critères et deux alternatives, tous aux noms différents.", icon=":material/error:")
        st.stop()
    exemple = tuple(criteres) == ex.AHP_CRITERES and tuple(alternatives) == ex.AHP_ALTERNATIVES

    st.markdown("**Comparaison des critères**")
    A_crit = saisie_paires(f"ahpc_c_{'_'.join(criteres)}", criteres, ex.AHP_MATRICE_CRITERES if exemple else None)
    matrices = []
    colonnes = st.columns(min(len(criteres), 3))
    for k, critere in enumerate(criteres):
        with colonnes[k % len(colonnes)]:
            st.markdown(f"**Alternatives selon : {critere}**")
            defaut = ex.AHP_MATRICES_ALTERNATIVES.get(critere) if exemple else None
            matrices.append(saisie_paires(f"ahpc_a_{k}_{'_'.join(alternatives)}", alternatives, defaut))
    res = executer(po.ahp_complet, A_crit, matrices)
    rangs = cl.rangs_depuis_scores(res["priorites_finales"])
    resultat_principal(alternatives, rangs, f"Priorité finale : {res['priorites_finales'].max():.4f}.")
    gr.barres(alternatives, res["priorites_finales"], "Priorité finale")

    table = pd.DataFrame(res["priorites_locales"], index=alternatives, columns=criteres)
    table["Priorité finale"] = res["priorites_finales"]
    table.loc["Poids du critère"] = [*res["criteres"]["poids"], np.nan]
    tableau_nombres(table)

    consistances = [("Critères", res["criteres"]["cr"])] + [
        (f"Selon {c}", r["cr"]) for c, r in zip(criteres, res["locales"])
    ]
    defauts = [nom for nom, cr in consistances if cr >= 0.1]
    if defauts:
        st.warning("Matrices à revoir (CR ≥ 0,1) : " + ", ".join(defauts) + ".", icon=":material/warning:")
    else:
        st.success("Toutes les matrices sont consistantes (CR < 0,1).", icon=":material/check_circle:")
    st.caption(" · ".join(f"{nom} : CR = {cr:.3f}" for nom, cr in consistances))
