import pandas as pd
import streamlit as st

from interface import graphiques as gr
from interface.commun import executer, saisie_paires, saisie_probleme, tableau_nombres
from mcdm import exemples as ex
from mcdm import ponderation as po

methode = st.segmented_control(
    "Méthode", ["AHP", "BWM", "Entropie", "CRITIC", "Poids hybrides"], default="AHP", key="pond_methode"
) or "AHP"

if methode == "AHP":
    st.caption("Exemple : achat d'une voiture. Sélectionnez une valeur par paire de critères.")
    noms_texte = st.text_input("Critères, séparés par des virgules", ", ".join(ex.AHP_CRITERES), key="ahp_noms")
    noms = [c.strip() for c in noms_texte.split(",") if c.strip()]
    if len(noms) < 2 or len(set(noms)) != len(noms):
        st.error("Indiquez au moins deux critères aux noms différents.", icon=":material/error:")
        st.stop()
    defaut = ex.AHP_MATRICE_CRITERES if tuple(noms) == ex.AHP_CRITERES else None
    A = saisie_paires(f"ahp_paires_{'_'.join(noms)}", noms, defaut)
    res = executer(po.ahp, A)
    if res["consistant"]:
        st.success(f"Matrice consistante : CR = {res['cr']:.4f} < 0,1. Les poids peuvent être conservés.", icon=":material/check_circle:")
    else:
        st.warning(f"Matrice inconsistante : CR = {res['cr']:.4f} ≥ 0,1. Revoyez les comparaisons.", icon=":material/warning:")
    gr.barres(noms, res["poids"], "Poids")
    a, b, c, d = st.columns(4)
    a.metric("λmax", f"{res['lambda_max']:.4f}")
    b.metric("CI", f"{res['ci']:.4f}")
    c.metric("RI", f"{res['ri']:.2f}")
    d.metric("CR", f"{res['cr']:.4f}")
    with st.expander("Détail des calculs"):
        st.markdown("Matrice de comparaison et somme de chaque colonne :")
        complete = pd.DataFrame(A, index=noms, columns=noms)
        complete.loc["Somme"] = res["sommes"]
        tableau_nombres(complete)
        st.markdown("Chaque case est divisée par la somme de sa colonne, puis on fait la moyenne de chaque ligne :")
        normalisee = pd.DataFrame(res["normalisee"], index=noms, columns=noms)
        normalisee["Priorité"] = res["poids"]
        tableau_nombres(normalisee)
        st.latex(r"\lambda_{max}=\sum_i s_i w_i,\quad CI=\frac{\lambda_{max}-n}{n-1},\quad CR=\frac{CI}{RI}")
        exact, lam_exact = po.ahp_vecteur_propre_exact(A)
        st.caption(
            f"Pour comparaison, le vecteur propre exact donne λmax = {lam_exact:.4f} et "
            f"les poids {', '.join(f'{v:.4f}' for v in exact)}."
        )

elif methode == "BWM":
    st.caption(
        "Exemple : choix d'un fournisseur. Une note de 1 désigne le meilleur "
        "critère (colonne BO) et le pire (colonne OW)."
    )
    table = st.data_editor(
        pd.DataFrame(
            {"Critère": ex.BWM_CRITERES, "Meilleur vers autres (BO)": ex.BWM_BO, "Autres vers pire (OW)": ex.BWM_OW}
        ),
        hide_index=True,
        num_rows="dynamic",
        column_config={
            c: st.column_config.NumberColumn(c, min_value=1, max_value=9, step=1)
            for c in ("Meilleur vers autres (BO)", "Autres vers pire (OW)")
        },
        key="bwm_table",
    ).dropna()
    noms = [str(c) for c in table["Critère"]]
    res = executer(po.bwm, table["Meilleur vers autres (BO)"], table["Autres vers pire (OW)"])
    st.success(
        f"Meilleur critère : **{noms[res['meilleur']]}**. Pire critère : **{noms[res['pire']]}**. "
        f"Écart ξ* = {res['xi']:.4f} : plus il est proche de 0, plus les jugements sont cohérents.",
        icon=":material/check_circle:",
    )
    if not res["note_max_sur_pire"]:
        st.warning("Contrôle de bon sens : la note la plus élevée du vecteur BO devrait porter sur le pire critère.", icon=":material/warning:")
    gr.barres(noms, res["poids"], "Poids")
    st.caption(f"BWM : {res['nb_comparaisons']} comparaisons, contre {res['nb_comparaisons_ahp']} pour AHP.")
    with st.expander("Modèle résolu"):
        st.latex(
            r"\min \xi \ \text{ s.c. } |w_B-a_{Bj}w_j|\le\xi,\ |w_j-a_{jW}w_W|\le\xi,\ \sum_j w_j=1,\ w_j\ge 0"
        )
        st.markdown("Contraintes sous la forme $Ax \\le 0$, avec $x=(w_1,\\dots,w_n,\\xi)$ :")
        tableau_nombres(pd.DataFrame(res["A_ub"], columns=[*noms, "ξ"]), decimales=0, hide_index=True)

elif methode == "Entropie":
    d = saisie_probleme("entropie", ex.MACHINES_CRITIC.titre)
    res = executer(po.entropie, d.X, d.sens)
    gr.barres(d.criteres, res["poids"], "Poids")
    tableau_nombres(
        pd.DataFrame(
            {"Entropie Ej": res["entropie"], "Diversification 1 − Ej": res["diversification"], "Poids": res["poids"]},
            index=d.criteres,
        )
    )
    with st.expander("Détail des calculs"):
        st.latex(r"p_{ij}=\frac{d_{ij}}{\sum_i d_{ij}},\quad E_j=-\frac{1}{\ln m}\sum_i p_{ij}\ln p_{ij},\quad w_j=\frac{1-E_j}{n-\sum_k E_k}")
        st.markdown("Matrice normalisée :")
        tableau_nombres(pd.DataFrame(res["normalisee"], index=d.alternatives, columns=d.criteres))
        st.markdown("Probabilités $p_{ij}$ :")
        tableau_nombres(pd.DataFrame(res["p"], index=d.alternatives, columns=d.criteres))

elif methode == "CRITIC":
    d = saisie_probleme("critic", ex.MACHINES_CRITIC.titre)
    res = executer(po.critic, d.X, d.sens)
    gr.barres(d.criteres, res["poids"], "Poids")
    tableau_nombres(
        pd.DataFrame({"Écart type σj": res["ecart_type"], "Indice Cj": res["indice_c"], "Poids": res["poids"]}, index=d.criteres)
    )
    st.markdown("**Corrélations entre critères**")
    gr.carte_chaleur(pd.DataFrame(res["correlation"], index=d.criteres, columns=d.criteres), "ρ", domaine=(-1, 1))
    with st.expander("Détail des calculs"):
        st.latex(r"r_{ij}=\frac{x_{ij}-\min x_j}{\max x_j-\min x_j}\ (+),\quad r_{ij}=\frac{\max x_j-x_{ij}}{\max x_j-\min x_j}\ (-)")
        st.latex(r"C_j=\sigma_j\sum_k\left(1-|\rho_{jk}|\right),\quad w_j=\frac{C_j}{\sum_j C_j}")
        tableau_nombres(pd.DataFrame(res["normalisee"], index=d.alternatives, columns=d.criteres))

else:
    d = saisie_probleme("hybride", ex.MACHINES_CRITIC.titre)
    alpha = st.slider("Part des poids subjectifs (α)", 0.0, 1.0, 0.5, 0.05, key="hybride_alpha")
    objectif = st.segmented_control("Poids objectifs", ["CRITIC", "Entropie"], default="CRITIC", key="hybride_obj") or "CRITIC"
    w_obj = executer(po.critic if objectif == "CRITIC" else po.entropie, d.X, d.sens)["poids"]
    w = po.poids_hybrides(d.poids, w_obj, alpha)
    st.caption("Les poids subjectifs sont ceux du tableau des critères ci-dessus.")
    long = pd.concat(
        [
            pd.DataFrame({"Critère": d.criteres, "Série": nom, "Poids": valeurs})
            for nom, valeurs in (("Subjectifs", d.poids), (objectif, w_obj), ("Hybrides", w))
        ]
    )
    gr.barres_groupees(long, "Critère", "Série", "Poids", "Poids")
