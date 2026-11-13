"""Nombres flous intuitionnistes et opérateur IFWA, utilisés par TOPSIS intuitionniste (diapos 112-117)."""
import numpy as np


# --- Ensembles flous intuitionnistes (diapos 112-117) ------------------------


def verifier_ifn(a):
    mu, nu = a
    if not (0 <= mu <= 1 and 0 <= nu <= 1 and mu + nu <= 1 + 1e-9):
        raise ValueError(f"Nombre intuitionniste invalide ({mu}, {nu}) : il faut μ + ν ≤ 1.")
    return a


def hesitation_ifn(a):
    return max(1 - a[0] - a[1], 0.0)


def ifn_somme(a, b):
    return (a[0] + b[0] - a[0] * b[0], a[1] * b[1])


def ifn_produit(a, b):
    return (a[0] * b[0], a[1] + b[1] - a[1] * b[1])


def ifn_scalaire(lam, a):
    return (1 - (1 - a[0]) ** lam, a[1] ** lam)


def ifn_puissance(a, lam):
    return (a[0] ** lam, 1 - (1 - a[1]) ** lam)


def ifwa(nombres, poids):
    """Opérateur IFWA (diapo 116)."""
    w = np.asarray(poids, dtype=float)
    if not np.isclose(w.sum(), 1, atol=1e-3):
        raise ValueError("Les poids σ doivent sommer à 1.")
    m = np.array([n[0] for n in nombres], dtype=float)
    v = np.array([n[1] for n in nombres], dtype=float)
    return (float(1 - np.prod((1 - m) ** w)), float(np.prod(v**w)))


def ifn_score(a):
    return a[0] - a[1]


def ifn_precision(a):
    return a[0] + a[1]


def ifn_comparer(a, b, tol=1e-12):
    """1 si a > b, -1 si a < b, 0 si égaux (score puis précision)."""
    sa, sb = ifn_score(a), ifn_score(b)
    if abs(sa - sb) > tol:
        return 1 if sa > sb else -1
    ha, hb = ifn_precision(a), ifn_precision(b)
    if abs(ha - hb) > tol:
        return 1 if ha > hb else -1
    return 0
