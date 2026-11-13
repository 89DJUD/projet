"""Graphiques Altair partagés. Une alternative garde sa couleur, quel que soit son rang."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Ordre validé pour la séparation entre couleurs, y compris pour les daltoniens.
COULEURS = ["#7c3aed", "#ec4899", "#e8930c", "#0d9488", "#2f6fdb", "#5f9e1b", "#a21caf", "#e11d48"]
MAUVE = COULEURS[0]
ROSE = COULEURS[1]
GRIS = "#b9a8c9"
ENCRE = "#2d1640"
ENCRE_DOUCE = "#5b4670"
SEQUENTIELLE = ["#f8f1fd", "#c4a1f2", "#5b21b6"]
DIVERGENTE = ["#db2777", "#f6f1f7", "#6d28d9"]


def _echelle_entites(noms):
    noms = list(noms)
    return alt.Scale(domain=noms, range=COULEURS[: len(noms)])


def barres(noms, valeurs, titre_valeur, tri_croissant=False, hauteur_ligne=30):
    """Barres horizontales triées : une seule série, donc pas de légende."""
    df = pd.DataFrame({"Nom": list(noms), "Valeur": np.asarray(valeurs, dtype=float)})
    # Ordre explicite : avec des couches filtrées, un tri sur l'axe x serait ignoré.
    ordre = list(df.sort_values("Valeur", ascending=tri_croissant, kind="stable")["Nom"])
    base = alt.Chart(df).encode(
        y=alt.Y("Nom:N", sort=ordre, title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        x=alt.X("Valeur:Q", title=titre_valeur, axis=alt.Axis(grid=True, gridOpacity=0.4, tickCount=5)),
        tooltip=["Nom", alt.Tooltip("Valeur:Q", format=".4f", title=titre_valeur)],
    )
    marques = base.mark_bar(cornerRadiusEnd=4, height=max(hauteur_ligne - 12, 8)).encode(
        color=alt.condition("datum.Valeur >= 0", alt.value(MAUVE), alt.value(ROSE))
    )
    texte = alt.Text("Valeur:Q", format=".3f")
    droite = base.transform_filter("datum.Valeur >= 0").mark_text(align="left", dx=5, color=ENCRE_DOUCE).encode(text=texte)
    gauche = base.transform_filter("datum.Valeur < 0").mark_text(align="right", dx=-5, color=ENCRE_DOUCE).encode(text=texte)
    st.altair_chart((marques + droite + gauche).properties(height=alt.Step(hauteur_ligne)))


def barres_groupees(df, categorie, serie, valeur, titre_valeur):
    """df au format long ; une couleur par série, avec légende."""
    series = list(dict.fromkeys(df[serie]))
    graphique = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3, size=12)
        .encode(
            y=alt.Y(f"{categorie}:N", sort=None, title=None),
            yOffset=alt.YOffset(f"{serie}:N", sort=series),
            x=alt.X(f"{valeur}:Q", title=titre_valeur, axis=alt.Axis(tickCount=5, gridOpacity=0.4)),
            color=alt.Color(f"{serie}:N", scale=_echelle_entites(series), legend=alt.Legend(title=None, orient="top")),
            tooltip=[categorie, serie, alt.Tooltip(f"{valeur}:Q", format=".3f")],
        )
        .properties(height=alt.Step(14))
    )
    st.altair_chart(graphique)


def trajectoires_rangs(table_rangs):
    """Rang de chaque alternative selon chaque méthode (lignes reliées). Au plus 8 alternatives."""
    df = table_rangs.reset_index(names="Alternative").melt("Alternative", var_name="Méthode", value_name="Rang")
    methodes = list(table_rangs.columns)
    base = alt.Chart(df).encode(
        x=alt.X("Méthode:N", sort=methodes, title=None, axis=alt.Axis(labelAngle=0, grid=True, gridOpacity=0.3)),
        y=alt.Y(
            "Rang:Q",
            scale=alt.Scale(reverse=True, domain=[0.7, table_rangs.to_numpy().max() + 0.3], nice=False),
            axis=alt.Axis(tickMinStep=1, format="d", grid=False),
        ),
        color=alt.Color(
            "Alternative:N",
            scale=_echelle_entites(table_rangs.index),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=["Alternative", "Méthode", "Rang"],
    )
    graphique = base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=90, opacity=1, stroke="white", strokeWidth=2)
    st.altair_chart(graphique.properties(height=280))


def carte_chaleur(df, titre_couleur, domaine=None, format_valeur=".2f"):
    """Matrice annotée : le texte change de teinte pour rester lisible sur les cases foncées.

    Un domaine centré sur 0 (corrélations) reçoit l'échelle divergente rose - gris - mauve.
    """
    lignes, colonnes = list(df.index), list(df.columns)
    long = df.reset_index(names="Ligne").melt("Ligne", var_name="Colonne", value_name="Valeur")
    vmin, vmax = domaine if domaine else (float(np.nanmin(df.to_numpy())), float(np.nanmax(df.to_numpy())))
    milieu = (vmin + vmax) / 2
    divergente = vmin < 0 < vmax and np.isclose(milieu, 0)
    if divergente:
        texte_clair = "abs(datum.Valeur) >= 0.6"
    else:
        texte_clair = f"datum.Valeur >= {vmin + 0.7 * (vmax - vmin)}"
    base = alt.Chart(long).encode(
        x=alt.X("Colonne:N", sort=colonnes, title=None, axis=alt.Axis(labelAngle=0, orient="top", domain=False, ticks=False, labelLimit=170)),
        y=alt.Y("Ligne:N", sort=lignes, title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=220)),
        tooltip=["Ligne", "Colonne", alt.Tooltip("Valeur:Q", format=format_valeur)],
    )
    cases = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(
            "Valeur:Q",
            title=titre_couleur,
            scale=alt.Scale(domain=[vmin, milieu, vmax], range=DIVERGENTE if divergente else SEQUENTIELLE),
            legend=alt.Legend(orient="right", gradientLength=120),
        )
    )
    textes = base.mark_text(fontSize=13).encode(
        text=alt.Text("Valeur:Q", format=format_valeur),
        color=alt.condition(texte_clair, alt.value("white"), alt.value(ENCRE)),
    )
    st.altair_chart((cases + textes).properties(height=max(36 * len(lignes) + 30, 120)))


def balayage(valeurs_w, scores, noms, titre_critere, poids_actuel, titre_score):
    """Score de chaque alternative selon le poids d'un critère ; trait vertical au poids actuel."""
    lignes = []
    for w, ligne in zip(valeurs_w, scores):
        for nom, s in zip(noms, ligne):
            lignes.append({"Poids": w, "Alternative": nom, "Score": float(s)})
    df = pd.DataFrame(lignes)
    courbes = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Poids:Q", title=f"Poids de « {titre_critere} »", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=5, gridOpacity=0.4)),
            y=alt.Y("Score:Q", title=titre_score, axis=alt.Axis(gridOpacity=0.4)),
            color=alt.Color("Alternative:N", scale=_echelle_entites(noms), legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("Poids:Q", format=".2f"), "Alternative", alt.Tooltip("Score:Q", format=".3f")],
        )
    )
    repere = (
        alt.Chart(pd.DataFrame({"Poids": [float(poids_actuel)]}))
        .mark_rule(color=ENCRE_DOUCE, strokeDash=[4, 4])
        .encode(x="Poids:Q", tooltip=[alt.Tooltip("Poids:Q", format=".3f", title="Poids actuel")])
    )
    st.altair_chart((courbes + repere).properties(height=300))


def courbes_appartenance(xs, courbes, titre_x="x", repere=None):
    """courbes : dictionnaire nom -> degrés d'appartenance."""
    df = pd.DataFrame({"x": np.tile(xs, len(courbes))})
    df["Ensemble"] = np.repeat(list(courbes), len(xs))
    df["μ"] = np.concatenate([np.asarray(v, dtype=float) for v in courbes.values()])
    couleur = alt.Color(
        "Ensemble:N",
        scale=_echelle_entites(courbes),
        legend=alt.Legend(title=None, orient="top") if len(courbes) > 1 else None,
    )
    graphique = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("x:Q", title=titre_x, scale=alt.Scale(domain=[float(min(xs)), float(max(xs))])),
            y=alt.Y("μ:Q", title="Degré d'appartenance", scale=alt.Scale(domain=[0, 1.05]), axis=alt.Axis(gridOpacity=0.4)),
            color=couleur,
            tooltip=[alt.Tooltip("x:Q", format=".2f"), "Ensemble", alt.Tooltip("μ:Q", format=".3f")],
        )
    )
    if repere is not None:
        graphique += (
            alt.Chart(pd.DataFrame({"x": [float(repere)]})).mark_rule(color=ENCRE_DOUCE, strokeDash=[4, 4]).encode(x="x:Q")
        )
    st.altair_chart(graphique.properties(height=260))


def courbe_roc(lignes, point_retenu=None):
    df = pd.DataFrame(lignes).sort_values(["FPR", "TPR"])
    axe_x = alt.X("FPR:Q", title="Taux de faux positifs (FPR)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=5, gridOpacity=0.4))
    axe_y = alt.Y("TPR:Q", title="Taux de vrais positifs (TPR)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=5, gridOpacity=0.4))
    diagonale = (
        alt.Chart(pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]}))
        .mark_line(color=GRIS, strokeDash=[4, 4])
        .encode(x=axe_x, y=axe_y)
    )
    infobulle = [
        alt.Tooltip("seuil:Q", format=".3f", title="Seuil τ"),
        alt.Tooltip("FPR:Q", format=".2f"),
        alt.Tooltip("TPR:Q", format=".2f"),
    ]
    courbe = alt.Chart(df).mark_line(color=MAUVE, strokeWidth=2).encode(x=axe_x, y=axe_y)
    points = alt.Chart(df).mark_point(color=MAUVE, filled=True, size=80).encode(x=axe_x, y=axe_y, tooltip=infobulle)
    graphique = diagonale + courbe + points
    if point_retenu:
        graphique += (
            alt.Chart(pd.DataFrame([point_retenu]))
            .mark_point(color=ROSE, filled=True, size=200, stroke="white", strokeWidth=2, opacity=1)
            .encode(x=axe_x, y=axe_y, tooltip=infobulle)
        )
    st.altair_chart(graphique.properties(height=320))


def evolution_series(df, x, y, serie, titre_y, facette=None, inverser=False):
    echelle_y = alt.Scale(reverse=True, zero=False) if inverser else alt.Scale(zero=False)
    axe_y = alt.Axis(gridOpacity=0.4, tickMinStep=1) if inverser else alt.Axis(gridOpacity=0.4)
    noms = list(dict.fromkeys(df[serie]))
    graphique = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, interpolate="step-after" if inverser else "linear")
        .encode(
            x=alt.X(f"{x}:Q", title="Période"),
            y=alt.Y(f"{y}:Q", title=titre_y, scale=echelle_y, axis=axe_y),
            color=alt.Color(f"{serie}:N", scale=_echelle_entites(noms), legend=alt.Legend(title=None, orient="top")),
            tooltip=[x, serie, alt.Tooltip(f"{y}:Q", format=".3f")],
        )
    )
    if facette:
        ordre_facettes = list(dict.fromkeys(df[facette]))
        graphique = graphique.properties(height=170, width=210).facet(
            facet=alt.Facet(f"{facette}:N", title=None, sort=ordre_facettes), columns=3
        ).resolve_scale(y="independent")
    else:
        graphique = graphique.properties(height=230)
    st.altair_chart(graphique)
