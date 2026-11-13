"""Classement des alternatives : WSM, WPM, WASPAS, TOPSIS, VIKOR."""
import numpy as np
from scipy.stats import rankdata


def rangs_depuis_scores(scores, plus_grand_meilleur=True):
    """Rang 1 = meilleure alternative ; les ex æquo partagent le même rang."""
    scores = np.asarray(scores, dtype=float)
    return rankdata(-scores if plus_grand_meilleur else scores, method="min").astype(int)


def _verifier(X, poids, sens):
    X = np.asarray(X, dtype=float)
    poids = np.asarray(poids, dtype=float)
    sens = np.asarray(sens)
    if X.ndim != 2 or X.shape[1] != poids.size or poids.size != sens.size:
        raise ValueError("Dimensions incohérentes entre la matrice, les poids et les sens.")
    if not np.isfinite(X).all():
        raise ValueError("La matrice de décision contient des cases vides.")
    if not np.isclose(poids.sum(), 1, atol=1e-3):
        raise ValueError(f"La somme des poids doit valoir 1 (elle vaut {poids.sum():.3f}).")
    return X, poids, sens


def normalisation_lineaire(X, sens):
    """Diapo 55 : r = x / max (bénéfice), r = min / x (coût)."""
    X = np.asarray(X, dtype=float)
    R = np.zeros_like(X)
    for j, s in enumerate(sens):
        col = X[:, j]
        if s > 0:
            if col.max() <= 0:
                raise ValueError(f"Critère {j + 1} : le maximum doit être positif.")
            R[:, j] = col / col.max()
        else:
            if (col <= 0).any():
                raise ValueError(f"Critère {j + 1} (coût) : toutes les valeurs doivent être > 0.")
            R[:, j] = col.min() / col
    return R


def wsm(X, poids, sens):
    X, poids, sens = _verifier(X, poids, sens)
    R = normalisation_lineaire(X, sens)
    scores = R @ poids
    return {"normalisee": R, "scores": scores, "rangs": rangs_depuis_scores(scores)}


def wpm(X, poids, sens):
    X, poids, sens = _verifier(X, poids, sens)
    R = normalisation_lineaire(X, sens)
    if (R < 0).any():
        raise ValueError("WPM exige des valeurs positives : un produit de puissances n'est pas défini ici.")
    scores = np.prod(np.power(R, poids), axis=1)
    return {"normalisee": R, "scores": scores, "rangs": rangs_depuis_scores(scores)}


def waspas(X, poids, sens, lam=0.5):
    """Diapo 57 : Q = λ·WSM + (1 - λ)·WPM."""
    q1 = wsm(X, poids, sens)
    q2 = wpm(X, poids, sens)
    scores = lam * q1["scores"] + (1 - lam) * q2["scores"]
    return {
        "normalisee": q1["normalisee"],
        "q1": q1["scores"],
        "q2": q2["scores"],
        "scores": scores,
        "rangs": rangs_depuis_scores(scores),
    }


def topsis(X, poids, sens):
    """Diapos 62-64."""
    X, poids, sens = _verifier(X, poids, sens)
    normes = np.sqrt((X**2).sum(axis=0))
    R = np.divide(X, normes, out=np.zeros_like(X), where=normes != 0)
    V = R * poids
    ideal = np.where(sens > 0, V.max(axis=0), V.min(axis=0))
    anti_ideal = np.where(sens > 0, V.min(axis=0), V.max(axis=0))
    s_plus = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    s_moins = np.sqrt(((V - anti_ideal) ** 2).sum(axis=1))
    total = s_plus + s_moins
    scores = np.divide(s_moins, total, out=np.zeros_like(total), where=total != 0)
    return {
        "normalisee": R,
        "ponderee": V,
        "ideal": ideal,
        "anti_ideal": anti_ideal,
        "s_plus": s_plus,
        "s_moins": s_moins,
        "scores": scores,
        "rangs": rangs_depuis_scores(scores),
    }


def vikor(X, poids, sens, v=0.5):
    """Diapos 66-71 : plus S, R et Q sont petits, meilleure est l'alternative."""
    X, poids, sens = _verifier(X, poids, sens)
    m = X.shape[0]
    meilleur = np.where(sens > 0, X.max(axis=0), X.min(axis=0))
    pire = np.where(sens > 0, X.min(axis=0), X.max(axis=0))
    ecart = meilleur - pire
    regrets = np.divide(meilleur - X, ecart, out=np.zeros_like(X), where=ecart != 0) * poids
    S = regrets.sum(axis=1)
    R = regrets.max(axis=1)

    def _echelle(val):
        e = val.max() - val.min()
        return (val - val.min()) / e if e > 0 else np.zeros_like(val)

    Q = v * _echelle(S) + (1 - v) * _echelle(R)
    ordre = np.argsort(Q, kind="stable")
    premier = int(ordre[0])
    dq = 1 / (m - 1) if m > 1 else 1.0
    avantage = m < 2 or Q[ordre[1]] - Q[premier] >= dq - 1e-12
    # Diapo 70 : A' doit aussi être la meilleure selon S et selon R.
    stabilite = bool(np.isclose(S[premier], S.min()) and np.isclose(R[premier], R.min()))
    if avantage and stabilite:
        compromis = [premier]
    elif avantage:
        compromis = [premier, int(ordre[1])]
    else:
        compromis = [int(i) for i in ordre if Q[i] - Q[premier] < dq - 1e-12]
    return {
        "meilleur": meilleur,
        "pire": pire,
        "regrets": regrets,
        "S": S,
        "R": R,
        "Q": Q,
        "scores": Q,
        "rangs": rangs_depuis_scores(Q, plus_grand_meilleur=False),
        "avantage_acceptable": bool(avantage),
        "stabilite_acceptable": stabilite,
        "seuil_dq": dq,
        "compromis": compromis,
    }


METHODES = {
    "WSM": wsm,
    "WPM": wpm,
    "WASPAS": waspas,
    "TOPSIS": topsis,
    "VIKOR": vikor,
}


def classer_toutes_methodes(X, poids, sens, noms=None):
    """Applique les méthodes demandées ; une méthode inapplicable est signalée, pas masquée."""
    rangs, scores, erreurs = {}, {}, {}
    for nom in noms or METHODES:
        try:
            res = METHODES[nom](X, poids, sens)
            rangs[nom] = res["rangs"]
            scores[nom] = res["scores"]
        except ValueError as exc:
            erreurs[nom] = str(exc)
    return rangs, scores, erreurs
