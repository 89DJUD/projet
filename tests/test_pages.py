"""Exécute chaque page dans le simulateur de Streamlit, dans chacune de ses vues."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

RACINE = Path(__file__).resolve().parent.parent
VUES = {
    "ponderation.py": ("pond_methode", ["AHP", "BWM", "Entropie", "CRITIC", "Poids hybrides"]),
    "additives.py": ("add_methode", ["WSM, WPM, WASPAS", "TOPSIS", "VIKOR", "AHP complet"]),
    "robustesse.py": ("rob_vue", ["Spearman : comparer des classements", "Sensibilité aux poids"]),
    "alternative_unique.py": ("uni_vue", ["Score MAUT et décision", "Choisir le seuil (ROC)"]),
    "mcdm_flou.py": ("mf_vue", ["Score flou d'une alternative", "TOPSIS flou", "TOPSIS intuitionniste"]),
}
SANS_VUE = ["simple.py", "decision.py"]
TOUTES_LES_PAGES = ["accueil.py", *SANS_VUE, *VUES]


def lancer(page, etat=None):
    at = AppTest.from_file(str(RACINE / "app_pages" / page), default_timeout=120)
    for cle, valeur in (etat or {}).items():
        at.session_state[cle] = valeur
    if page == "decision.py":
        for cle, valeur in {"dec_methode_poids": "Saisis à la main", "dec_alpha": 0.5, "dec_lambda": 0.5, "dec_v": 0.5}.items():
            if cle not in (etat or {}):
                at.session_state[cle] = valeur
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", SANS_VUE)
def test_page_simple(page):
    lancer(page)


@pytest.mark.parametrize("page", TOUTES_LES_PAGES)
def test_navigation_depuis_le_point_d_entree(page):
    at = AppTest.from_file(str(RACINE / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    at.switch_page(f"app_pages/{page}").run()
    assert not at.exception, [e.value for e in at.exception]


def test_etat_de_l_application_finale_conserve_entre_les_pages():
    """Une valeur modifiée dans l'application finale survit à une visite d'une autre page."""
    at = AppTest.from_file(str(RACINE / "streamlit_app.py"), default_timeout=120).run()
    at.switch_page("app_pages/decision.py").run()
    at.slider(key="dec_lambda").set_value(0.8).run()
    at.switch_page("app_pages/accueil.py").run()
    at.switch_page("app_pages/decision.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.slider(key="dec_lambda").value == 0.8


@pytest.mark.parametrize(
    "page,cle,vue", [(p, c, v) for p, (c, vues) in VUES.items() for v in vues]
)
def test_page_avec_vues(page, cle, vue):
    at = lancer(page, {cle: vue})
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("methode", ["Saisis à la main", "Égaux", "AHP", "BWM", "Entropie", "CRITIC", "Hybride"])
def test_decision_chaque_methode_de_poids(methode):
    at = lancer("decision.py", {"dec_methode_poids": methode})
    assert not at.error, [e.value for e in at.error]
    assert any("Recommandation" in s.value for s in [*at.success, *at.info, *at.warning])


def test_decision_exemple_a_valeurs_negatives():
    """Sur Vuković et al., WPM et WASPAS sont signalées comme inapplicables, le reste fonctionne."""
    from mcdm import exemples as ex

    at = lancer("decision.py")
    at.selectbox(key="dec_exemple").set_value(ex.ACTIONS_VUKOVIC.titre).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    avertissements = " ".join(w.value for w in at.warning)
    assert "WPM" in avertissements and "WASPAS" in avertissements


def test_decision_simple_suit_les_criteres_et_chaque_methode():
    import pandas as pd

    at = lancer("simple.py")
    assert "M" in at.success[0].value
    criteres = at.session_state["simple_criteres__valeur"]
    ajout = pd.concat([criteres, pd.DataFrame({"Critère": ["Confort"], "Sens": ["Bénéfice (+)"], "Poids": [0.1]})])
    at.session_state["simple_criteres__valeur"] = ajout.reset_index(drop=True)
    at.session_state["simple_criteres__version"] = 1
    at.run()
    assert "Confort" in at.session_state["simple_matrice__valeur"].columns
    assert any("Remplissez" in e.value for e in at.error)  # la nouvelle colonne est vide
    for methode in ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]:
        at = lancer("simple.py", {"simple_methode": methode})
        assert at.success and not at.error


@pytest.mark.parametrize("vue", ["WSM, WPM, WASPAS", "TOPSIS", "VIKOR"])
@pytest.mark.parametrize("poids", ["Saisis à la main", "Égaux", "AHP", "BWM", "Entropie", "CRITIC", "Hybride"])
def test_additives_chaque_ponderation(vue, poids):
    at = lancer("additives.py", {"add_methode": vue, "add_poids": poids})
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("poids", ["Saisis à la main", "Égaux", "AHP", "BWM", "Entropie", "CRITIC", "Hybride"])
def test_decision_simple_chaque_ponderation(poids):
    at = lancer("simple.py", {"simple_poids": poids})
    assert at.success and not at.error, [e.value for e in at.error]
