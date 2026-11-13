"""Analyse multicritère floue : score flou, TOPSIS flou (Chen, 2000) et TOPSIS intuitionniste."""
import numpy as np

from mcdm.classement import rangs_depuis_scores
from mcdm.flou import ifwa

ECHELLE_NOTES = {
    "VP": (0, 0, 1),
    "P": (0, 1, 3),
    "MP": (1, 3, 5),
    "F": (3, 5, 7),
    "MG": (5, 7, 9),
    "G": (7, 9, 10),
    "VG": (9, 10, 10),
}

ECHELLE_POIDS = {
    "VL": (0, 0, 0.1),
    "L": (0, 0.1, 0.3),
    "ML": (0.1, 0.3, 0.5),
    "M": (0.3, 0.5, 0.7),
    "MH": (0.5, 0.7, 0.9),
    "H": (0.7, 0.9, 1),
    "VH": (0.9, 1, 1),
}

LIBELLES_NOTES = {
    "VP": "Very poor",
    "P": "Poor",
    "MP": "Medium poor",
    "F": "Fair",
    "MG": "Medium good",
    "G": "Good",
    "VG": "Very good",
}

LIBELLES_POIDS = {
    "VL": "Very low",
    "L": "Low",
    "ML": "Medium low",
    "M": "Medium",
    "MH": "Medium high",
    "H": "High",
    "VH": "Very high",
}

# Le cours ne donne pas d'échelle intuitionniste : échelle d'illustration, à adapter.
ECHELLE_NOTES_IF = {
    "VP": (0.05, 0.90),
    "P": (0.20, 0.70),
    "MP": (0.35, 0.55),
    "F": (0.50, 0.40),
    "MG": (0.65, 0.25),
    "G": (0.80, 0.10),
    "VG": (0.95, 0.03),
}

ECHELLE_POIDS_IF = {
    "VL": (0.05, 0.90),
    "L": (0.20, 0.70),
    "ML": (0.35, 0.55),
    "M": (0.50, 0.40),
    "MH": (0.65, 0.25),
    "H": (0.80, 0.10),
    "VH": (0.95, 0.03),
}


def defuzzifier(t):
    """Centre de gravité d'un nombre triangulaire (a + b + c) / 3."""
    return float(np.mean(t))


def score_flou(evaluations, poids, maximums):
    """Exemple de la diapo 134 : normaliser, agréger, défuzzifier une seule alternative."""
    D = np.asarray(evaluations, dtype=float)
    w = np.asarray(poids, dtype=float)
    max_c = np.asarray(maximums, dtype=float)
    if (max_c <= 0).any():
        raise ValueError("Les maximums de normalisation doivent être positifs.")
    R = D / max_c[:, None]
    agrege = (R * w[:, None]).sum(axis=0)
    return {
        "normalisee": R,
        "agrege": agrege,
        "score": defuzzifier(agrege),
        "incertitude": float(agrege[2] - agrege[0]),
    }


def _verifier_triangles(T, nom):
    if not (T[..., 0] <= T[..., 1]).all() or not (T[..., 1] <= T[..., 2]).all():
        raise ValueError(f"{nom} : chaque triplet (a, b, c) doit vérifier a ≤ b ≤ c.")


def topsis_flou(notes, poids, sens=None):
    """TOPSIS flou de Chen (diapos 135-139).

    notes : tableau (alternatives, décideurs, critères, 3) ; poids : (décideurs, critères, 3).
    Pour un critère de coût, la diapo 136 écrit (x¹⁻/x¹, x¹⁻/x², x¹⁻/x³) : ce triplet serait
    décroissant, on prend donc l'ordre de Chen, (x¹⁻/x³, x¹⁻/x², x¹⁻/x¹).
    """
    notes = np.asarray(notes, dtype=float)
    poids = np.asarray(poids, dtype=float)
    _verifier_triangles(notes, "Notes")
    _verifier_triangles(poids, "Poids")
    m, _, n, _ = notes.shape
    sens = np.ones(n) if sens is None else np.asarray(sens)

    X = notes.mean(axis=1)
    W = poids.mean(axis=0)
    R = np.zeros_like(X)
    for j in range(n):
        if sens[j] > 0:
            maximum = X[:, j, 2].max()
            if maximum <= 0:
                raise ValueError(f"Critère {j + 1} : le maximum doit être positif.")
            R[:, j] = X[:, j] / maximum
        else:
            minimum = X[:, j, 0].min()
            if (X[:, j] <= 0).any():
                raise ValueError(f"Critère {j + 1} (coût) : les notes collectives doivent être > 0.")
            R[:, j] = minimum / X[:, j, ::-1]
    V = R * W[None, :, :]
    d_plus_detail = np.sqrt(((V - 1) ** 2).sum(axis=2) / 3)
    d_moins_detail = np.sqrt((V**2).sum(axis=2) / 3)
    d_plus = d_plus_detail.sum(axis=1)
    d_moins = d_moins_detail.sum(axis=1)
    cc = d_moins / (d_moins + d_plus)
    return {
        "collective": X,
        "poids": W,
        "normalisee": R,
        "ponderee": V,
        "d_plus": d_plus,
        "d_moins": d_moins,
        "scores": cc,
        "rangs": rangs_depuis_scores(cc),
    }


def topsis_intuitionniste(notes, poids, sens=None, poids_decideurs=None):
    """TOPSIS flou intuitionniste (diapos 143-150).

    notes : (alternatives, décideurs, critères, 2) ; poids : (décideurs, critères, 2).
    """
    notes = np.asarray(notes, dtype=float)
    poids = np.asarray(poids, dtype=float)
    m, l, n, _ = notes.shape
    if (notes.sum(axis=-1) > 1 + 1e-9).any() or (poids.sum(axis=-1) > 1 + 1e-9).any():
        raise ValueError("Chaque nombre intuitionniste doit vérifier μ + ν ≤ 1.")
    sigma = np.full(l, 1 / l) if poids_decideurs is None else np.asarray(poids_decideurs, dtype=float)
    sens = np.ones(n) if sens is None else np.asarray(sens)

    D = np.zeros((m, n, 2))
    for i in range(m):
        for j in range(n):
            D[i, j] = ifwa([tuple(notes[i, k, j]) for k in range(l)], sigma)
    W = np.array([ifwa([tuple(poids[k, j]) for k in range(l)], sigma) for j in range(n)])

    # a ⊗ w = (μa μw, νa + νw - νa νw)
    P = np.zeros_like(D)
    P[..., 0] = D[..., 0] * W[:, 0]
    P[..., 1] = D[..., 1] + W[:, 1] - D[..., 1] * W[:, 1]
    pi = np.clip(1 - P[..., 0] - P[..., 1], 0, 1)

    mu_plus = np.where(sens > 0, P[..., 0].max(axis=0), P[..., 0].min(axis=0))
    nu_plus = np.where(sens > 0, P[..., 1].min(axis=0), P[..., 1].max(axis=0))
    mu_moins = np.where(sens > 0, P[..., 0].min(axis=0), P[..., 0].max(axis=0))
    nu_moins = np.where(sens > 0, P[..., 1].max(axis=0), P[..., 1].min(axis=0))
    pi_plus = 1 - mu_plus - nu_plus
    pi_moins = 1 - mu_moins - nu_moins

    s_plus = np.sqrt(
        (P[..., 0] - mu_plus) ** 2 + (P[..., 1] - nu_plus) ** 2 + (pi - pi_plus) ** 2
    ).sum(axis=1) / (2 * n)
    s_moins = np.sqrt(
        (P[..., 0] - mu_moins) ** 2 + (P[..., 1] - nu_moins) ** 2 + (pi - pi_moins) ** 2
    ).sum(axis=1) / (2 * n)
    total = s_plus + s_moins
    c = np.divide(s_moins, total, out=np.zeros_like(total), where=total != 0)
    return {
        "agregee": D,
        "poids": W,
        "ponderee": P,
        "hesitation": pi,
        "ideal": np.column_stack([mu_plus, nu_plus]),
        "anti_ideal": np.column_stack([mu_moins, nu_moins]),
        "s_plus": s_plus,
        "s_moins": s_moins,
        "scores": c,
        "rangs": rangs_depuis_scores(c),
    }


def notes_vers_tableau(notes_ling, echelle):
    """Convertit une liste imbriquée de libellés en tableau numérique via une échelle."""
    return np.array(
        [[[echelle[x] for x in ligne] for ligne in bloc] for bloc in notes_ling], dtype=float
    )


def poids_vers_tableau(poids_ling, echelle):
    return np.array([[echelle[x] for x in ligne] for ligne in poids_ling], dtype=float)
