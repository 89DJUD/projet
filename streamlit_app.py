from pathlib import Path

import streamlit as st

from interface.style import appliquer_style

st.set_page_config(page_title="Aide à la décision", page_icon=":material/alt_route:", layout="wide")
appliquer_style()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

# Réglages de la décision complète à conserver quand on visite une autre page.
REGLAGES_DECISION = {"dec_methode_poids": "Saisis à la main", "dec_alpha": 0.5, "dec_lambda": 0.5, "dec_v": 0.5}
for cle, defaut in REGLAGES_DECISION.items():
    st.session_state[cle] = st.session_state.get(cle, defaut)
for cle in ("dec_exemple", "dec_sensib_methode", "dec_sensib_critere"):
    if cle in st.session_state:
        st.session_state[cle] = st.session_state[cle]

page = st.navigation(
    {
        "": [
            st.Page("app_pages/simple.py", title="Décision simple", icon=":material/table_chart:", default=True),
            st.Page("app_pages/accueil.py", title="Accueil", icon=":material/home:"),
            st.Page("app_pages/decision.py", title="Décision complète", icon=":material/rocket_launch:"),
        ],
        "Méthodes classiques": [
            st.Page("app_pages/ponderation.py", title="Pondération des critères", icon=":material/balance:"),
            st.Page("app_pages/additives.py", title="Méthodes additives", icon=":material/functions:"),
            st.Page("app_pages/robustesse.py", title="Robustesse et comparaison", icon=":material/verified:"),
            st.Page("app_pages/alternative_unique.py", title="Une seule alternative", icon=":material/rule:"),
        ],
        "Logique floue": [
            st.Page("app_pages/mcdm_flou.py", title="MCDM floue", icon=":material/tune:"),
        ],
    }
)

# L'accueil a son propre bandeau.
if page.title != "Accueil":
    st.title(page.title)
page.run()
