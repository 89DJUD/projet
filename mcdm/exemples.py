"""Jeux de données du cours (exercices des diapos) et petites données d'illustration."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Probleme:
    titre: str
    source: str
    alternatives: tuple
    criteres: tuple
    sens: tuple  # +1 : à maximiser, -1 : à minimiser
    matrice: tuple
    poids: tuple

    @property
    def X(self):
        return np.array(self.matrice, dtype=float)


MACHINES_WASPAS = Probleme(
    "Choix d'une machine (5 critères)",
    "Exercice, diapo 58",
    ("A1", "A2", "A3", "A4"),
    ("C1", "C2", "C3", "C4", "C5"),
    (-1, -1, 1, -1, 1),
    (
        (0.035, 847, 0.335, 1.760, 0.590),
        (0.027, 834, 0.335, 1.680, 0.665),
        (0.037, 808, 0.590, 2.400, 0.500),
        (0.028, 821, 0.500, 1.590, 0.410),
    ),
    (0.331, 0.181, 0.369, 0.072, 0.047),
)

VOITURES_TOPSIS = Probleme(
    "Choix d'une voiture (4 marques)",
    "Exercice, diapo 65",
    ("M1", "M2", "M3", "M4"),
    ("Style", "Fiabilité", "Économie de carburant", "Coût"),
    (1, 1, 1, -1),
    ((7, 9, 9, 8), (8, 7, 8, 7), (9, 6, 8, 9), (6, 7, 8, 6)),
    (0.2, 0.1, 0.4, 0.3),
)

MACHINES_VIKOR = Probleme(
    "Choix d'une machine (2 critères)",
    "Exercice, diapo 72",
    ("A1", "A2", "A3"),
    ("C1", "C2"),
    (-1, 1),
    ((1, 3000), (2, 3750), (5, 4500)),
    (0.5, 0.5),
)

MACHINES_CRITIC = Probleme(
    "Machines de découpage",
    "Exercice, diapo 51",
    ("A1", "A2", "A3", "A4"),
    ("Épaisseur max", "Largeur de coupe", "Qualité de découpe", "Coût de maintenance"),
    (-1, -1, 1, -1),
    ((30, 0.1, 1, 20), (100, 0.7, 1, 40), (50, 1, 2, 10), (300, 2, 3, 35)),
    (0.25, 0.25, 0.25, 0.25),
)

ACTIONS_VUKOVIC = Probleme(
    "Sélection d'actions (18 titres)",
    "Vuković et al., 2020, diapo 18",
    (
        "ADPL", "ADRS2", "ARNT", "ATGR", "ATPL", "DDJH", "DLKV", "ERNT", "HT",
        "INGR", "KOEI", "KRAS", "MAIS", "OPTE", "PODR", "RIVP", "VLEN", "ZABA",
    ),
    ("Mean", "Std. dev.", "Volume", "Beta", "EPS", "ROE", "P/B", "P/S", "ROA"),
    (1, -1, 1, -1, 1, 1, -1, -1, 1),
    (
        (0.2137, 2.19, 1382.93, 0.51, 16.72, 9.36, 0.95, 0.66, 5.20),
        (-0.1037, 1.94, 2302.18, 0.68, 55.08, 3.63, 0.28, 0.67, 1.90),
        (-0.3592, 2.67, 864.13, 0.37, 17.17, 5.63, 1.47, 3.24, 3.29),
        (0.2753, 3.06, 420.82, 1.12, 82.84, 12.28, 1.28, 0.54, 5.39),
        (0.1222, 5.90, 762.22, 0.06, 58.38, 14.42, 1.58, 2.44, 5.70),
        (-1.2446, 7.06, 4504.48, 0.68, -3.29, -38.62, 2.29, 0.38, -5.35),
        (-0.5443, 5.65, 10656.87, 1.42, 0.50, 5.23, 1.98, 0.32, 0.94),
        (-0.1648, 2.69, 162.01, 0.70, 50.98, 28.82, 6.81, 1.08, 8.26),
        (-0.1194, 1.60, 7220.56, 0.46, 9.85, 6.40, 1.07, 1.74, 5.11),
        (0.1794, 5.22, 13159.67, 1.77, 0.95, 12.02, 0.46, 0.72, 1.50),
        (-0.2807, 2.34, 374.91, 0.81, 42.03, 4.35, 0.74, 0.65, 2.91),
        (-0.3532, 2.44, 194.47, 0.79, 22.17, 4.57, 0.95, 0.62, 2.51),
        (0.1381, 2.79, 311.65, 1.33, 14.82, 10.33, 1.95, 2.78, 5.50),
        (-0.4341, 5.26, 28452.85, 1.18, -0.85, -303.61, 9.70, 0.38, -8.96),
        (-0.0184, 2.54, 1419.53, 1.03, 3.35, 0.82, 0.66, 0.47, 0.47),
        (-0.0424, 2.66, 31328.13, 0.70, 1.94, 9.74, 2.19, 3.14, 4.91),
        (-0.4791, 8.55, 25731.26, 1.87, 2.18, 14.86, 0.94, 0.48, 9.08),
        (-0.0144, 3.73, 6927.05, 1.47, 3.26, 5.76, 0.93, 3.11, 0.82),
    ),
    (0.0370, 0.0189, 0.0533, 0.0259, 0.2182, 0.3070, 0.0764, 0.1543, 0.1089),
)

PROBLEMES = {p.titre: p for p in (VOITURES_TOPSIS, MACHINES_WASPAS, MACHINES_VIKOR, MACHINES_CRITIC, ACTIONS_VUKOVIC)}

# --- AHP : achat d'une voiture (diapos 32-34) ---------------------------------

AHP_CRITERES = ("Coût", "Confort", "Sécurité")
AHP_ALTERNATIVES = ("V1", "V2")
AHP_MATRICE_CRITERES = ((1, 7, 3), (1 / 7, 1, 1 / 3), (1 / 3, 3, 1))
AHP_MATRICES_ALTERNATIVES = {
    "Coût": ((1, 7), (1 / 7, 1)),
    "Confort": ((1, 1 / 5), (5, 1)),
    "Sécurité": ((1, 1 / 9), (9, 1)),
}

# --- BWM : choix d'un fournisseur (diapo 37) ---------------------------------

BWM_CRITERES = ("Qualité", "Prix", "Délai de livraison", "Engagement RSE")
BWM_BO = (1, 3, 4, 8)
BWM_OW = (8, 5, 3, 1)

# --- Spearman : classements de la diapo 78 ------------------------------------

SPEARMAN_ALTERNATIVES = ("A1", "A2", "A3", "A4", "A5", "A6")
SPEARMAN_RANGS = {
    "TOPSIS": (2, 1, 3, 5, 4, 6),
    "VIKOR": (1, 2, 4, 3, 5, 6),
}

# --- Choix du seuil : décisions passées de la diapo 85 ------------------------

SEUIL_ALTERNATIVES = ("A1", "A2", "A3", "A4", "A5", "A6")
SEUIL_SCORES = (0.85, 0.78, 0.71, 0.69, 0.62, 0.54)
SEUIL_DECISIONS = (1, 1, 1, 0, 0, 0)

# --- TOPSIS flou : recrutement (diapos 140-142) -------------------------------

RECRUTEMENT_CRITERES = (
    "Stabilité émotionnelle",
    "Communication orale",
    "Personnalité",
    "Expérience passée",
    "Confiance en soi",
)
RECRUTEMENT_ALTERNATIVES = ("A1", "A2", "A3")
RECRUTEMENT_DECIDEURS = ("DM1", "DM2", "DM3")
RECRUTEMENT_NOTES = (
    (("MG", "G", "F", "VG", "F"), ("G", "MG", "G", "G", "F"), ("MG", "F", "G", "VG", "F")),
    (("G", "VG", "VG", "VG", "VG"), ("G", "VG", "VG", "VG", "MG"), ("MG", "VG", "G", "VG", "G")),
    (("VG", "MG", "G", "G", "G"), ("G", "G", "MG", "VG", "G"), ("F", "VG", "VG", "MG", "MG")),
)
RECRUTEMENT_POIDS = (
    ("H", "VH", "VH", "VH", "M"),
    ("VH", "VH", "H", "VH", "MH"),
    ("MH", "VH", "H", "VH", "MH"),
)
