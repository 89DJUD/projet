"""Saisie et affichage partagés entre les pages."""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
import streamlit as st

from mcdm import exemples as ex

SENS_LIBELLES = {1: "Bénéfice (+)", -1: "Coût (−)"}
SENS_VALEURS = {v: k for k, v in SENS_LIBELLES.items()}
ECHELLE_SAATY = [str(Fraction(1, k)) for k in range(9, 1, -1)] + [str(k) for k in range(1, 10)]
CLE_PROBLEME_APP = "Mon problème (décision complète)"


@dataclass
class Donnees:
    alternatives: list
    criteres: list
    sens: np.ndarray
    X: np.ndarray
    poids: np.ndarray


def editeur_persistant(cle, defaut, **options):
    """st.data_editor dont le contenu survit à un changement de page.

    Streamlit efface l'état d'un widget absent de la page : à chaque retour, on repart de la
    dernière version enregistrée. La base reste figée tant que le widget existe, car la modifier
    réinitialiserait les éditions en cours.
    """
    version = st.session_state.get(f"{cle}__version", 0)
    cle_widget = f"{cle}__v{version}"
    if cle_widget not in st.session_state or f"{cle}__base" not in st.session_state:
        st.session_state[f"{cle}__base"] = st.session_state.get(f"{cle}__valeur", defaut)
    edite = st.data_editor(st.session_state[f"{cle}__base"], key=cle_widget, **options)
    st.session_state[f"{cle}__valeur"] = edite
    return edite


def remplacer_editeur(cle, valeur):
    """Force un nouveau contenu : à appeler avant le rendu du widget, ou dans un rappel."""
    st.session_state[f"{cle}__valeur"] = valeur
    st.session_state.pop(f"{cle}__base", None)
    st.session_state[f"{cle}__version"] = st.session_state.get(f"{cle}__version", 0) + 1


def stop_si_erreur(message):
    st.error(message, icon=":material/error:")
    st.stop()


def normaliser_poids(poids):
    poids = np.asarray(poids, dtype=float)
    if not np.isfinite(poids).all() or (poids < 0).any() or poids.sum() <= 0:
        stop_si_erreur("Les poids doivent être positifs ou nuls, avec une somme strictement positive.")
    if abs(poids.sum() - 1) > 1e-3:
        st.caption(f"La somme des poids valait {poids.sum():.3f} : ils ont été ramenés à 1.")
    return poids / poids.sum()


def probleme_vers_tables(pb):
    criteres = pd.DataFrame(
        {
            "Critère": list(pb.criteres),
            "Sens": [SENS_LIBELLES[s] for s in pb.sens],
            "Poids": list(pb.poids),
        }
    )
    matrice = pd.DataFrame(pb.X, columns=list(pb.criteres))
    matrice.insert(0, "Alternative", list(pb.alternatives))
    return criteres, matrice


def tables_vers_donnees(criteres, matrice):
    """Valide la saisie et arrête la page avec un message clair si elle est inexploitable."""
    criteres = criteres.dropna(subset=["Critère"])
    noms_criteres = [str(c) for c in criteres["Critère"]]
    matrice = matrice.dropna(subset=["Alternative"])
    alternatives = [str(a) for a in matrice["Alternative"]]
    if len(noms_criteres) < 2 or len(alternatives) < 2:
        stop_si_erreur("Il faut au moins deux alternatives et deux critères.")
    if len(set(noms_criteres)) != len(noms_criteres) or len(set(alternatives)) != len(alternatives):
        stop_si_erreur("Les noms des alternatives et des critères doivent être uniques.")
    if criteres["Sens"].isna().any():
        stop_si_erreur("Indiquez le sens (bénéfice ou coût) de chaque critère.")
    valeurs = matrice[noms_criteres].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(valeurs).all():
        stop_si_erreur("Complétez toutes les cases de la matrice de décision avec des nombres.")
    sens = np.array([SENS_VALEURS[s] for s in criteres["Sens"]])
    return Donnees(alternatives, noms_criteres, sens, valeurs, criteres["Poids"].to_numpy(dtype=float))


CONFIG_CRITERES = {
    "Sens": st.column_config.SelectboxColumn("Sens", options=list(SENS_VALEURS), required=True),
    "Poids": st.column_config.NumberColumn("Poids", min_value=0.0, format="%.4f"),
}


def saisie_probleme(cle, defaut):
    """Choix d'un jeu de données d'exemple, modifiable sur place, pour les pages de méthodes."""
    options = list(ex.PROBLEMES)
    if "pb_criteres__valeur" in st.session_state:
        options.append(CLE_PROBLEME_APP)
    choix = st.selectbox("Jeu de données", options, index=options.index(defaut), key=f"{cle}_jeu")
    if choix == CLE_PROBLEME_APP:
        criteres = st.session_state["pb_criteres__valeur"]
        matrice = st.session_state["pb_matrice__valeur"]
    else:
        pb = ex.PROBLEMES[choix]
        st.caption("Toutes les cases sont modifiables.")
        criteres, matrice = probleme_vers_tables(pb)
    criteres = st.data_editor(
        criteres,
        hide_index=True,
        disabled=["Critère"],
        column_config=CONFIG_CRITERES,
        key=f"{cle}_crit_{choix}",
    )
    matrice = st.data_editor(
        matrice,
        hide_index=True,
        num_rows="dynamic",
        key=f"{cle}_mat_{choix}",
    )
    donnees = tables_vers_donnees(criteres, matrice)
    donnees.poids = normaliser_poids(donnees.poids)
    return donnees


def matrice_depuis_paires(noms, defaut=None):
    """Prépare le tableau des comparaisons par paires (triangle supérieur) sur l'échelle de Saaty."""
    lignes = []
    for i, a in enumerate(noms):
        for j, b in enumerate(noms):
            if j > i:
                valeur = "1"
                if defaut is not None:
                    v = float(defaut[i][j])
                    valeur = str(Fraction(v).limit_denominator(9))
                lignes.append({"Élément A": a, "Élément B": b, "Importance de A par rapport à B": valeur})
    return pd.DataFrame(lignes)


def paires_vers_matrice(noms, table):
    n = len(noms)
    A = np.eye(n)
    indices = {nom: k for k, nom in enumerate(noms)}
    for _, ligne in table.iterrows():
        i, j = indices[ligne["Élément A"]], indices[ligne["Élément B"]]
        v = float(Fraction(ligne["Importance de A par rapport à B"]))
        A[i, j], A[j, i] = v, 1 / v
    return A


def saisie_paires(cle, noms, defaut=None, persistant=False):
    """Seules n(n-1)/2 comparaisons sont saisies : la réciprocité est garantie par construction."""
    table = matrice_depuis_paires(noms, defaut)
    options = dict(
        hide_index=True,
        disabled=["Élément A", "Élément B"],
        column_config={
            "Importance de A par rapport à B": st.column_config.SelectboxColumn(
                "Importance de A par rapport à B",
                options=ECHELLE_SAATY,
                required=True,
                help="1 = égale, 3 = modérée, 5 = forte, 7 = très forte, 9 = extrême ; 1/k si B l'emporte.",
            )
        },
    )
    if persistant:
        sauvegarde = st.session_state.get(f"{cle}__valeur")
        if sauvegarde is not None and list(sauvegarde["Élément A"]) != list(table["Élément A"]):
            remplacer_editeur(cle, table)
        elif sauvegarde is not None and list(sauvegarde["Élément B"]) != list(table["Élément B"]):
            remplacer_editeur(cle, table)
        edite = editeur_persistant(cle, table, **options)
    else:
        edite = st.data_editor(table, key=cle, **options)
    return paires_vers_matrice(noms, edite)


def tableau_nombres(df, decimales=4, **options):
    """Affiche un DataFrame avec un nombre fixe de décimales."""
    config = {
        c: st.column_config.NumberColumn(format=f"%.{decimales}f")
        for c in df.columns
        if pd.api.types.is_float_dtype(df[c])
    }
    st.dataframe(df, column_config=config, **options)


def table_classement(alternatives, scores, rangs, titre_score="Score"):
    df = pd.DataFrame({"Alternative": alternatives, titre_score: scores, "Rang": rangs}).sort_values(
        ["Rang", "Alternative"]
    )
    tableau_nombres(df.set_index("Alternative"))


def resultat_principal(alternatives, rangs, phrase_score=""):
    """Annonce la ou les meilleures alternatives en tête de section."""
    meilleurs = [a for a, r in zip(alternatives, rangs) if r == 1]
    texte = " et ".join(f"**{a}**" for a in meilleurs)
    verbe = "arrive" if len(meilleurs) == 1 else "arrivent"
    st.success(f"{texte} {verbe} en tête. {phrase_score}".strip(), icon=":material/emoji_events:")


def executer(fonction, *args, **kwargs):
    """Transforme une erreur de validation du moteur en message lisible."""
    try:
        return fonction(*args, **kwargs)
    except ValueError as exc:
        stop_si_erreur(str(exc))


METHODES_POIDS = {
    "Saisis à la main": "Les poids du tableau des critères ci-dessus.",
    "Égaux": "Aucune préférence entre critères.",
    "AHP": "Comparaisons par paires et test de consistance.",
    "BWM": "Comparaisons au meilleur et au pire critère.",
    "Entropie": "Objectif : favorise les critères dispersés.",
    "CRITIC": "Objectif : favorise les critères dispersés et peu corrélés.",
    "Hybride": "Combine un avis d'expert (AHP) et les données (CRITIC).",
}


def _poids_ahp(cle, noms):
    from mcdm import ponderation as po

    res = executer(po.ahp, saisie_paires(f"{cle}_ahp_{'_'.join(noms)}", noms))
    if res["consistant"]:
        st.success(f"Matrice consistante : CR = {res['cr']:.3f} < 0,1.", icon=":material/check_circle:")
    else:
        st.warning(f"Matrice inconsistante : CR = {res['cr']:.3f} ≥ 0,1. Revoyez vos comparaisons.", icon=":material/warning:")
    return res["poids"]


def choix_poids(cle, donnees):
    """Laisse choisir les poids : saisis à la main ou calculés par une méthode de pondération."""
    from mcdm import ponderation as po

    noms, X, sens, n = donnees.criteres, donnees.X, donnees.sens, len(donnees.criteres)
    methode = st.pills("Pondération des critères", list(METHODES_POIDS), default="Saisis à la main", key=f"{cle}_poids") or "Saisis à la main"
    st.caption(METHODES_POIDS[methode])
    if methode == "Saisis à la main":
        poids = donnees.poids
    elif methode == "Égaux":
        poids = np.full(n, 1 / n)
    elif methode == "AHP":
        poids = _poids_ahp(cle, noms)
    elif methode == "BWM":
        table = st.data_editor(
            pd.DataFrame(
                {
                    "Critère": noms,
                    "Meilleur vers autres (BO)": [1] + [2] * (n - 2) + [3],
                    "Autres vers pire (OW)": [3] + [2] * (n - 2) + [1],
                }
            ),
            key=f"{cle}_bwm_{'_'.join(noms)}",
            hide_index=True,
            disabled=["Critère"],
            column_config={
                c: st.column_config.NumberColumn(c, min_value=1, max_value=9, step=1)
                for c in ("Meilleur vers autres (BO)", "Autres vers pire (OW)")
            },
        )
        res = executer(po.bwm, table["Meilleur vers autres (BO)"], table["Autres vers pire (OW)"])
        poids = res["poids"]
        st.caption(f"Écart ξ* = {res['xi']:.4f} : plus il est proche de 0, plus les jugements sont cohérents.")
    elif methode == "Entropie":
        poids = executer(po.entropie, X, sens)["poids"]
    elif methode == "CRITIC":
        poids = executer(po.critic, X, sens)["poids"]
    else:
        alpha = st.slider("Part de l'avis d'expert (α)", 0.0, 1.0, 0.5, 0.05, key=f"{cle}_alpha", help="w = α · AHP + (1 − α) · CRITIC")
        poids = po.poids_hybrides(_poids_ahp(cle, noms), executer(po.critic, X, sens)["poids"], alpha)
    if methode != "Saisis à la main":
        st.dataframe(
            pd.DataFrame({"Critère": noms, "Poids": poids}),
            hide_index=True,
            column_config={"Poids": st.column_config.NumberColumn(format="%.4f")},
        )
    return np.asarray(poids, dtype=float)
