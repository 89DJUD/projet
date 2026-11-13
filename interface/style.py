"""Habillage rose et mauve : ce que le thème de config.toml ne permet pas (dégradés, ombres, bandeau)."""
import streamlit as st

CSS = """
<style>
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 420px at 100% -5%, rgba(236, 72, 153, 0.10), transparent 70%),
    radial-gradient(800px 400px at 15% -10%, rgba(124, 58, 237, 0.09), transparent 70%),
    #fffbfe;
}
[data-testid="stHeader"] { background: transparent; }

/* Barre latérale */
[data-testid="stSidebar"] { background: linear-gradient(175deg, #2e1250 0%, #4c1d7a 55%, #831a6d 100%); }
[data-testid="stSidebarNavLink"] { border-radius: 10px; transition: background 0.15s ease; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(249, 168, 212, 0.16); }
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: linear-gradient(90deg, rgba(236, 72, 153, 0.55), rgba(167, 139, 250, 0.35));
}
[data-testid="stSidebarNavLink"] span { color: #f7ecfb; }
[data-testid="stNavSectionHeader"] span { color: #f9a8d4; font-weight: 600; }

/* Titres : la page d'accueil a son propre bandeau */
.stApp h1 {
  background: linear-gradient(90deg, #6d28d9, #c026d3 55%, #db2777);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

/* Indicateurs */
[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid #ecdcf5;
  border-left: 4px solid #c026d3;
  border-radius: 14px;
  padding: 14px 18px;
  box-shadow: 0 4px 16px rgba(124, 58, 237, 0.07);
}
[data-testid="stMetricValue"] { color: #6d28d9; font-weight: 700; }

/* Boutons */
[data-testid="stBaseButton-secondary"] {
  border-color: #d8b4f0;
  color: #6d28d9;
  font-weight: 600;
  transition: all 0.15s ease;
}
[data-testid="stBaseButton-secondary"]:hover {
  background: linear-gradient(90deg, #7c3aed, #db2777);
  border-color: transparent;
  color: #ffffff;
}

/* Onglets et dépliants */
[data-baseweb="tab"] { font-weight: 600; }
[data-testid="stExpander"] details {
  background: #ffffff;
  box-shadow: 0 2px 10px rgba(124, 58, 237, 0.05);
}

/* Bandeau d'accueil */
.bandeau {
  background: linear-gradient(115deg, #5b21b6 0%, #a21caf 55%, #db2777 100%);
  border-radius: 22px;
  padding: 38px 42px 30px;
  color: #ffffff;
  box-shadow: 0 16px 40px rgba(162, 28, 175, 0.28);
}
.bandeau .sur-titre {
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #fbdcef;
}
.bandeau .grand-titre {
  font-family: Poppins, sans-serif;
  font-size: clamp(1.9rem, 4vw, 2.9rem);
  font-weight: 700;
  line-height: 1.15;
  margin: 8px 0 12px;
}
.bandeau .accroche { font-size: 1.05rem; line-height: 1.55; max-width: 720px; color: #fff4fb; }
.bandeau .chiffres { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 24px; }
.bandeau .chiffre {
  flex: 1 1 140px;
  background: rgba(255, 255, 255, 0.14);
  border: 1px solid rgba(255, 255, 255, 0.22);
  border-radius: 14px;
  padding: 12px 16px;
}
.bandeau .chiffre b { display: block; font-family: Poppins, sans-serif; font-size: 1.7rem; line-height: 1.2; }
.bandeau .chiffre span { font-size: 0.86rem; color: #fff0f9; }

/* Cartes d'accueil */
[class*="st-key-carte"] {
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 6px 22px rgba(124, 58, 237, 0.08);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[class*="st-key-carte"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 30px rgba(192, 38, 211, 0.16);
}
.st-key-carte_action [data-testid="stPageLink-NavLink"] {
  background: linear-gradient(90deg, #7c3aed, #db2777);
  border-radius: 999px;
  padding: 6px 18px;
  width: fit-content;
}
.st-key-carte_action [data-testid="stPageLink-NavLink"] * { color: #ffffff !important; font-weight: 600; }

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; }
}
</style>
"""


def appliquer_style():
    st.html(CSS)


def bandeau_accueil(chiffres):
    """chiffres : liste de (nombre, libellé) affichés sous l'accroche."""
    cases = "".join(f'<div class="chiffre"><b>{n}</b><span>{libelle}</span></div>' for n, libelle in chiffres)
    st.html(
        f"""
        <div class="bandeau">
          <div class="sur-titre">Aide à la décision multicritère</div>
          <div class="grand-titre">Décidez avec méthode,<br>comparez en confiance</div>
          <div class="accroche">
            Une seule plateforme pour pondérer vos critères, classer vos alternatives,
            tester la robustesse de vos choix et obtenir une recommandation justifiée.
          </div>
          <div class="chiffres">{cases}</div>
        </div>
        """
    )
