import streamlit as st

from interface.style import bandeau_accueil

# Méthodes réellement proposées dans l'application : les chiffres du bandeau en découlent.
METHODES = {
    "méthodes de pondération": ["AHP", "BWM", "Entropie", "CRITIC", "Poids hybrides"],
    "méthodes additives": ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR", "AHP complet"],
    "analyses de robustesse": ["Spearman", "Sensibilité aux poids", "Seuils d'inversion"],
    "méthodes MCDM floues": ["Score flou", "TOPSIS flou", "TOPSIS intuitionniste"],
}
bandeau_accueil([(len(noms), libelle) for libelle, noms in METHODES.items()])

with st.container(border=True, key="carte_action"):
    st.subheader("Prendre une décision de bout en bout", anchor=False)
    st.markdown(
        "Décrivez vos alternatives et vos critères, choisissez les poids, comparez cinq méthodes de "
        "classement, testez la robustesse et obtenez une recommandation justifiée, exportable en rapport."
    )
    st.page_link("app_pages/decision.py", label="Démarrer une décision", icon=":material/arrow_forward:")

st.subheader("Quelle méthode choisir ?", anchor=False)
col_questions, col_conseil = st.columns([2, 3], gap="large")
with col_questions:
    donnees = st.segmented_control(
        "Nature des données", ["Valeurs exactes", "Jugements imprécis"], default="Valeurs exactes"
    )
    decideurs = st.segmented_control("Décideurs", ["Un seul", "Plusieurs"], default="Un seul")
    avis = st.segmented_control(
        "Avis d'experts disponibles pour les poids ?", ["Beaucoup de temps", "Un peu", "Aucun"], default="Un peu"
    )
    nombre = st.segmented_control("Alternatives à évaluer", ["Plusieurs", "Une seule"], default="Plusieurs")

with col_conseil:
    conseils = []
    if avis == "Beaucoup de temps":
        conseils.append(("Poids", "AHP, avec ses comparaisons par paires et son test de consistance."))
    elif avis == "Un peu":
        conseils.append(("Poids", "BWM, qui demande 2n − 3 comparaisons seulement."))
    elif avis == "Aucun":
        conseils.append(("Poids", "CRITIC ou entropie, calculés à partir de la matrice de décision."))
    if nombre == "Une seule":
        conseils.append(("Évaluation", "MAUT et un seuil de décision, ou la distance à l'idéal."))
    elif donnees == "Jugements imprécis":
        if decideurs == "Plusieurs":
            conseils.append(("Classement", "TOPSIS flou, ou TOPSIS intuitionniste si les experts hésitent."))
        else:
            conseils.append(("Classement", "TOPSIS flou avec des variables linguistiques."))
    else:
        conseils.append(("Classement", "WSM, WASPAS, TOPSIS ou VIKOR."))
    conseils.append(("Toujours", "Comparez plusieurs méthodes et faites varier les poids avant de conclure."))
    with st.container(border=True, key="carte_conseil"):
        for titre, texte in conseils:
            st.markdown(f"**{titre} :** {texte}")

st.subheader("Contenu de la plateforme", anchor=False)
parties = [
    (
        "Méthodes classiques",
        [
            ("app_pages/ponderation.py", "Pondération", "AHP, BWM, entropie, CRITIC, poids hybrides"),
            ("app_pages/additives.py", "Additives", "WSM, WPM, WASPAS, TOPSIS, VIKOR, AHP complet"),
            ("app_pages/robustesse.py", "Robustesse", "Spearman, sensibilité aux poids"),
            ("app_pages/alternative_unique.py", "Une seule alternative", "MAUT, seuil et ROC, distance à l'idéal"),
        ],
    ),
    (
        "Logique floue",
        [
            ("app_pages/mcdm_flou.py", "MCDM floue", "Score flou, TOPSIS flou, TOPSIS intuitionniste"),
        ],
    ),
]
for k, (colonne, (titre, pages)) in enumerate(zip(st.columns(2), parties)):
    with colonne:
        with st.container(border=True, height="stretch", key=f"carte_partie_{k}"):
            st.markdown(f"**{titre}**")
            for chemin, libelle, detail in pages:
                st.page_link(chemin, label=libelle)
                st.caption(detail)
