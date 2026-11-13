from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from interface import graphiques as gr
from interface.commun import (
    CONFIG_CRITERES,
    SENS_LIBELLES,
    editeur_persistant,
    executer,
    normaliser_poids,
    probleme_vers_tables,
    remplacer_editeur,
    saisie_paires,
    stop_si_erreur,
    tableau_nombres,
    tables_vers_donnees,
)
from mcdm import classement as cl
from mcdm import exemples as ex
from mcdm import ponderation as po
from mcdm import robustesse as ro

METHODES_SENSIBILITE = {"TOPSIS": cl.topsis, "WSM": cl.wsm, "VIKOR": cl.vikor}


def charger_exemple(titre):
    criteres, matrice = probleme_vers_tables(ex.PROBLEMES[titre])
    remplacer_editeur("pb_criteres", criteres)
    remplacer_editeur("pb_matrice", matrice)


def lire_csv(fichier):
    """Première colonne : alternatives ; autres colonnes : critères, tous en bénéfice par défaut."""
    try:
        df = pd.read_csv(fichier, sep=None, engine="python", decimal=",")
        if df.shape[1] > 1 and not all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns[1:]):
            fichier.seek(0)
            df = pd.read_csv(fichier, sep=None, engine="python")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        stop_si_erreur(f"Lecture du fichier impossible : {exc}")
    df = df.rename(columns={df.columns[0]: "Alternative"})
    if df.shape[1] < 3:
        stop_si_erreur("Le fichier doit contenir une colonne d'alternatives et au moins deux critères.")
    noms = list(df.columns[1:])
    criteres = pd.DataFrame(
        {"Critère": noms, "Sens": SENS_LIBELLES[1], "Poids": 1 / len(noms)}
    )
    return criteres, df


if "pb_matrice__valeur" not in st.session_state:
    charger_exemple(ex.VOITURES_TOPSIS.titre)

onglet_pb, onglet_poids, onglet_rangs, onglet_robuste, onglet_reco = st.tabs(
    ["1. Problème", "2. Poids", "3. Classements", "4. Robustesse", "5. Recommandation"]
)

# --- Étape 1 : problème --------------------------------------------------------
with onglet_pb:
    st.markdown("Partez d'un exemple, importez un CSV, ou modifiez directement les tableaux.")
    col_exemple, col_csv = st.columns(2)
    with col_exemple:
        titre_exemple = st.selectbox("Exemple", list(ex.PROBLEMES), key="dec_exemple")
        st.button(
            "Charger cet exemple",
            icon=":material/download:",
            on_click=charger_exemple,
            args=(titre_exemple,),
        )
    with col_csv:
        fichier = st.file_uploader(
            "Ou importer un CSV",
            type="csv",
            help=(
                "Première colonne : noms des alternatives. Colonnes suivantes : un critère par colonne. "
                "Après l'import, tous les critères sont en bénéfice avec des poids égaux : corrigez-les ci-dessous."
            ),
        )
        if fichier is not None and st.button("Utiliser ce fichier", icon=":material/upload_file:"):
            criteres_csv, matrice_csv = lire_csv(fichier)
            remplacer_editeur("pb_criteres", criteres_csv)
            remplacer_editeur("pb_matrice", matrice_csv)
            st.rerun()

    st.markdown("**Critères**")
    criteres = editeur_persistant(
        "pb_criteres",
        None,
        hide_index=True,
        num_rows="dynamic",
        column_config=CONFIG_CRITERES,
    )

    # Les colonnes de la matrice suivent le tableau des critères.
    noms_criteres = [str(c) for c in criteres["Critère"].dropna()]
    matrice_actuelle = st.session_state["pb_matrice__valeur"]
    colonnes_actuelles = [c for c in matrice_actuelle.columns if c != "Alternative"]
    if noms_criteres != colonnes_actuelles and len(set(noms_criteres)) == len(noms_criteres):
        if len(noms_criteres) == len(colonnes_actuelles):
            nouvelle = matrice_actuelle.rename(columns=dict(zip(colonnes_actuelles, noms_criteres)))
        else:
            nouvelle = matrice_actuelle.reindex(columns=["Alternative", *noms_criteres])
        remplacer_editeur("pb_matrice", nouvelle)

    st.markdown("**Matrice de décision**")
    matrice = editeur_persistant("pb_matrice", None, hide_index=True, num_rows="dynamic")
    donnees = tables_vers_donnees(criteres, matrice)

alternatives, noms, sens, X = donnees.alternatives, donnees.criteres, donnees.sens, donnees.X
m, n = X.shape

# --- Étape 2 : poids -----------------------------------------------------------
FAMILLES = {
    "Saisis à la main": "Les poids du tableau des critères.",
    "Égaux": "Aucune préférence entre critères.",
    "AHP": "Comparaisons par paires et test de consistance.",
    "BWM": "Comparaisons au meilleur et au pire critère.",
    "Entropie": "Objectif : favorise les critères dispersés.",
    "CRITIC": "Objectif : favorise les critères dispersés et peu corrélés.",
    "Hybride": "Combine un avis d'expert (AHP) et les données (CRITIC).",
}


def poids_ahp_saisis():
    defaut = np.ones((n, n))
    matrice_ahp = saisie_paires("dec_ahp", noms, defaut, persistant=True)
    res = executer(po.ahp, matrice_ahp)
    if res["consistant"]:
        st.success(f"Matrice consistante : CR = {res['cr']:.3f} < 0,1.", icon=":material/check_circle:")
    else:
        st.warning(
            f"Matrice inconsistante : CR = {res['cr']:.3f} ≥ 0,1. Revoyez vos comparaisons avant de conclure.",
            icon=":material/warning:",
        )
    return res["poids"]


with onglet_poids:
    methode_poids = st.pills("Méthode de pondération", list(FAMILLES), key="dec_methode_poids") or "Saisis à la main"
    st.caption(FAMILLES[methode_poids])
    if methode_poids == "Saisis à la main":
        poids = normaliser_poids(donnees.poids)
    elif methode_poids == "Égaux":
        poids = np.full(n, 1 / n)
    elif methode_poids == "AHP":
        poids = poids_ahp_saisis()
    elif methode_poids == "BWM":
        table_bwm = editeur_persistant(
            f"dec_bwm_{'_'.join(noms)}",
            pd.DataFrame(
                {
                    "Critère": noms,
                    "Meilleur vers autres (BO)": [1] + [2] * (n - 2) + [3],
                    "Autres vers pire (OW)": [3] + [2] * (n - 2) + [1],
                }
            ),
            hide_index=True,
            disabled=["Critère"],
            column_config={
                c: st.column_config.NumberColumn(c, min_value=1, max_value=9, step=1)
                for c in ("Meilleur vers autres (BO)", "Autres vers pire (OW)")
            },
        )
        res_bwm = executer(po.bwm, table_bwm["Meilleur vers autres (BO)"], table_bwm["Autres vers pire (OW)"])
        poids = res_bwm["poids"]
        st.caption(
            f"Meilleur critère : {noms[res_bwm['meilleur']]}. Pire critère : {noms[res_bwm['pire']]}. "
            f"Écart ξ* = {res_bwm['xi']:.4f} : plus il est proche de 0, plus les jugements sont cohérents."
        )
    elif methode_poids == "Entropie":
        poids = executer(po.entropie, X, sens)["poids"]
    elif methode_poids == "CRITIC":
        poids = executer(po.critic, X, sens)["poids"]
    else:
        alpha = st.slider(
            "Part de l'avis d'expert (α)", 0.0, 1.0, step=0.05, key="dec_alpha",
            help="w = α · AHP + (1 − α) · CRITIC",
        )
        poids = po.poids_hybrides(poids_ahp_saisis(), executer(po.critic, X, sens)["poids"], alpha)

    gr.barres(noms, poids, "Poids")

# --- Étape 3 : classements -----------------------------------------------------
with onglet_rangs:
    col_lam, col_v = st.columns(2)
    lam = col_lam.slider("λ de WASPAS", 0.0, 1.0, step=0.1, key="dec_lambda", help="Part de WSM dans le score.")
    v = col_v.slider("v de VIKOR", 0.0, 1.0, step=0.1, key="dec_v", help="Poids de l'utilité de groupe.")

    fonctions = {
        "WSM": lambda: cl.wsm(X, poids, sens),
        "WPM": lambda: cl.wpm(X, poids, sens),
        "WASPAS": lambda: cl.waspas(X, poids, sens, lam),
        "TOPSIS": lambda: cl.topsis(X, poids, sens),
        "VIKOR": lambda: cl.vikor(X, poids, sens, v),
    }
    resultats, inapplicables = {}, {}
    for nom_methode, fonction in fonctions.items():
        try:
            resultats[nom_methode] = fonction()
        except ValueError as exc:
            inapplicables[nom_methode] = str(exc)
    if not resultats:
        stop_si_erreur("Aucune méthode n'a pu s'appliquer à ces données.")
    rangs = {k: r["rangs"] for k, r in resultats.items()}
    consensus = ro.agreger_rangs(rangs)
    ordre = np.argsort(consensus["rang_moyen"], kind="stable")

    table_rangs = pd.DataFrame(rangs, index=alternatives)
    affichage = table_rangs.iloc[ordre].copy()
    affichage.insert(0, "Consensus", consensus["rang_consensus"][ordre])
    st.markdown("**Rang de chaque alternative selon chaque méthode** (1 = la meilleure)")
    st.dataframe(affichage)
    for nom_methode, message in inapplicables.items():
        st.warning(f"{nom_methode} ne s'applique pas ici : {message}", icon=":material/warning:")

    if m <= 8:
        gr.trajectoires_rangs(table_rangs)
    else:
        st.caption("Plus de huit alternatives : le tableau remplace le graphique, qui deviendrait illisible.")

    with st.expander("Scores de chaque méthode"):
        lignes_scores = pd.DataFrame({k: r["scores"] for k, r in resultats.items()}, index=alternatives)
        tableau_nombres(lignes_scores)
        st.caption(
            "VIKOR : plus le score est petit, mieux c'est."
        )
    if "VIKOR" in resultats:
        res_v = resultats["VIKOR"]
        compromis = ", ".join(alternatives[i] for i in res_v["compromis"])
        st.info(
            f"**Compromis VIKOR : {compromis}.** "
            f"Avantage acceptable : {'oui' if res_v['avantage_acceptable'] else 'non'}. "
            f"Stabilité acceptable : {'oui' if res_v['stabilite_acceptable'] else 'non'}.",
            icon=":material/handshake:",
        )

# --- Étape 4 : robustesse ------------------------------------------------------
noms_methodes, matrice_rho = ro.matrice_spearman(rangs)
triangle = matrice_rho[np.triu_indices(len(noms_methodes), k=1)]
rho_moyen = float(np.nanmean(triangle)) if triangle.size else 1.0

with onglet_robuste:
    st.markdown("**Accord entre méthodes (corrélation de Spearman)**")
    gr.carte_chaleur(pd.DataFrame(matrice_rho, index=noms_methodes, columns=noms_methodes), "ρ", domaine=(-1, 1))
    critique = ro.rho_critique(m)
    if critique is not None and critique < 1:
        st.caption(f"Avec n = {m} alternatives, ρ est significatif à 5 % à partir de {critique}.")
    else:
        st.caption(f"Avec n = {m} alternatives, ces corrélations restent trop peu nombreuses pour être significatives.")

    # Après un changement de problème, l'ancien choix peut ne plus exister.
    if st.session_state.get("dec_sensib_critere") not in noms:
        st.session_state.pop("dec_sensib_critere", None)
    if st.session_state.get("dec_sensib_methode") not in rangs:
        st.session_state.pop("dec_sensib_methode", None)
    col_sens, col_crit = st.columns(2)
    methode_sens = col_sens.selectbox(
        "Méthode testée", [k for k in METHODES_SENSIBILITE if k in rangs], key="dec_sensib_methode"
    )
    critere_sens = col_crit.selectbox("Critère dont on fait varier le poids", noms, key="dec_sensib_critere")
    j = noms.index(critere_sens)
    fonction_sens = METHODES_SENSIBILITE[methode_sens]

    points = ro.balayage_poids(fonction_sens, X, poids, sens, j)
    trace = m <= 8
    if trace:
        gr.balayage(
            [p[0] for p in points],
            [p[1] for p in points],
            alternatives,
            critere_sens,
            poids[j],
            f"Score {methode_sens}" + (" (petit = mieux)" if methode_sens == "VIKOR" else ""),
        )
    meilleure, seuils = ro.seuil_inversion(fonction_sens, X, poids, sens)
    st.markdown(f"**Poids qui ferait perdre la première place à {alternatives[meilleure]}**")
    st.dataframe(
        pd.DataFrame(
            {
                "Critère": noms,
                "Poids actuel": np.round(poids, 3),
                "Poids d'inversion le plus proche": [
                    f"{s:.2f}" if s is not None else "Aucun" for s in seuils
                ],
            }
        ),
        hide_index=True,
    )

# --- Étape 5 : recommandation --------------------------------------------------
indice_reco = int(ordre[0])
reco = alternatives[indice_reco]
nb_premier = int(consensus["nb_premier"][indice_reco])
nb_methodes = consensus["nb_methodes"]

if nb_premier / nb_methodes >= 2 / 3:
    niveau, boite, icone = "élevée", st.success, ":material/verified:"
elif nb_premier / nb_methodes >= 0.5:
    niveau, boite, icone = "moyenne", st.info, ":material/info:"
else:
    niveau, boite, icone = "faible", st.warning, ":material/warning:"

with onglet_reco:
    boite(f"**Recommandation : {reco}.** Confiance {niveau}.", icon=icone)
    metrique_1, metrique_2 = st.columns(2)
    metrique_1.metric("Méthodes qui la classent première", f"{nb_premier} / {nb_methodes}")
    metrique_2.metric("Accord moyen entre méthodes (ρ)", f"{rho_moyen:.2f}")
    st.caption(
        "Règle de l'application, à ajuster selon l'enjeu. Confiance élevée : première pour au moins 2/3 des méthodes. "
        "Moyenne : au moins 1/2 des méthodes."
    )

    st.markdown("**Classement de consensus** (moyenne des rangs)")
    consensus_df = pd.DataFrame(
        {
            "Alternative": alternatives,
            "Rang de consensus": consensus["rang_consensus"],
            "Rang moyen": consensus["rang_moyen"],
            "Fois première": consensus["nb_premier"],
        }
    ).sort_values(["Rang de consensus", "Alternative"])
    tableau_nombres(consensus_df, decimales=2, hide_index=True)

    if niveau != "élevée":
        st.markdown(
            "**Avant de décider :** les méthodes ne convergent pas assez. "
            "Comparez les premières alternatives à la main et réexaminez les poids des critères les plus sensibles."
        )

    lignes_poids = "\n".join(f"| {c} | {SENS_LIBELLES[s]} | {w:.4f} |" for c, s, w in zip(noms, sens, poids))
    lignes_cons = "\n".join(
        f"| {r['Rang de consensus']} | {r['Alternative']} | {r['Rang moyen']:.2f} | {r['Fois première']} |"
        for _, r in consensus_df.iterrows()
    )
    colonnes_rangs = list(affichage.columns)
    table_rangs_md = "\n".join(
        [
            "| Alternative | " + " | ".join(colonnes_rangs) + " |",
            "|---" * (len(colonnes_rangs) + 1) + "|",
            *(
                f"| {nom} | " + " | ".join(str(v) for v in ligne) + " |"
                for nom, ligne in zip(affichage.index, affichage.to_numpy())
            ),
        ]
    )
    rapport = f"""# Rapport de décision

Date : {date.today():%d/%m/%Y}

## Recommandation

**{reco}**, avec une confiance {niveau} : première pour {nb_premier} méthodes sur {nb_methodes}, accord moyen entre méthodes ρ = {rho_moyen:.2f}.

## Critères et poids ({methode_poids})

| Critère | Sens | Poids |
|---|---|---|
{lignes_poids}

## Classement de consensus

| Rang | Alternative | Rang moyen | Fois première |
|---|---|---|---|
{lignes_cons}

## Rangs par méthode

{table_rangs_md}
"""
    col_rapport, col_csv_export = st.columns(2)
    col_rapport.download_button(
        "Télécharger le rapport (Markdown)",
        rapport,
        file_name="rapport_decision.md",
        mime="text/markdown",
        icon=":material/description:",
    )
    col_csv_export.download_button(
        "Télécharger les rangs (CSV)",
        affichage.to_csv().encode("utf-8-sig"),
        file_name="rangs_par_methode.csv",
        mime="text/csv",
        icon=":material/table_view:",
    )
