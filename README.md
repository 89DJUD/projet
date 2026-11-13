# Plateforme d'aide à la décision

Application finale du cours d'aide à la décision du Pr Lamrani Alaoui (EMI, 2026). Elle réunit les méthodes du cours et les enchaîne dans un parcours complet : problème, poids, classements, robustesse, recommandation.

## Lancer l'application

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Méthodes couvertes

| Partie | Méthodes | Page |
|---|---|---|
| Application finale | Parcours complet, consensus de 5 méthodes, rapport exportable | Application finale |
| Pondération | AHP, BWM, entropie, CRITIC, poids hybrides | Pondération des critères |
| Classement additif | WSM, WPM, WASPAS, TOPSIS, VIKOR, AHP complet | Méthodes additives |
| Robustesse | Spearman, balayage des poids, seuils d'inversion | Robustesse et comparaison |
| Une seule alternative | MAUT, seuil de décision (ROC, Youden, distance, coût), distance à l'idéal | Une seule alternative |
| MCDM floue | Score flou, TOPSIS flou (Chen), TOPSIS intuitionniste | MCDM floue |

## Organisation du code

- `mcdm/` : moteur de calcul, sans dépendance à Streamlit.
- `interface/` : saisie et graphiques partagés.
- `app_pages/` : une page par thème ; `streamlit_app.py` est le point d'entrée.
- `tests/` : vérifications contre le cours et simulation de chaque page.

```powershell
python -m pytest
```

## Vérification contre le cours

Les tests comparent le moteur aux résultats publiés :

| Référence | Résultat attendu | Statut |
|---|---|---|
| Notebook `Fuzzy_flou.ipynb` (TOPSIS flou) | CC = 0,6380 ; 0,7653 ; 0,7018 | Identique |
| Diapo 79 (Spearman TOPSIS–VIKOR) | 0,77 | Identique |
| Diapo 88 (seuils) | TPR, FPR, d et J des 3 cas | Identique |
| Diapo 134 (score flou) | (0,74 ; 0,85 ; 1,00), S = 0,86 | Identique |
| Diapo 42 (BWM) | ξ* = 0,08 | ξ* concorde (0,0838), pas les poids : voir ci-dessous |

Les exercices sans corrigé (AHP, CRITIC, WASPAS, TOPSIS, VIKOR) sont contrôlés par des calculs manuels partiels et des propriétés mathématiques.

## Écarts avec les diapos

| Diapo | Écrit dans le cours | Dans l'application | Raison |
|---|---|---|---|
| 42 | Poids (0,552 ; 0,207 ; 0,155 ; 0,086) | (0,5629 ; 0,2156 ; 0,1617 ; 0,0599) | Les poids de la diapo donnent \|w₂ − 5 w₄\| = 0,223, très au-dessus de ξ* = 0,08 : ils violent le modèle de la diapo 40 |
| 136 | Coût : (x¹⁻/x¹, x¹⁻/x², x¹⁻/x³) | (x¹⁻/x³, x¹⁻/x², x¹⁻/x¹) | Le triplet de la diapo est décroissant, ce n'est pas un nombre flou |
| 89 | Distances non pondérées | Écarts mis à l'échelle et pondérés | Sinon le critère aux plus grands chiffres écrase les autres |

## Ce qui ne vient pas du cours

- **TOPSIS intuitionniste :** l'échelle linguistique est une illustration, car le cours n'en donne pas.
- **Données d'illustration :** le dossier de crédit (MAUT) est fictif.

## Limites connues

- **MODM :** l'optimisation multi-objectifs est seulement définie dans le cours (diapo 7). Elle n'a pas de page.
