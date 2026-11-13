"""Pondération des critères : AHP, BWM (subjectives), entropie, CRITIC (objectives)."""
import numpy as np
from scipy.optimize import linprog

# Indice de consistance aléatoire (cours, diapo 29). n = 1 ou 2 : matrice toujours consistante.
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56}


def verifier_matrice_comparaison(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("La matrice de comparaison doit être carrée.")
    if not np.isfinite(A).all() or (A <= 0).any():
        raise ValueError("Les comparaisons doivent être des nombres strictement positifs.")
    if not np.allclose(A * A.T, 1, atol=1e-2):
        raise ValueError("La matrice doit être réciproque : a(j, i) = 1 / a(i, j).")
    return A


def ahp(A):
    """Méthode approximative du cours (diapos 26-29) : poids, λmax, CI et CR."""
    A = verifier_matrice_comparaison(A)
    n = A.shape[0]
    if n > 10:
        raise ValueError("La table RI s'arrête à n = 10.")
    sommes = A.sum(axis=0)
    normalisee = A / sommes
    poids = normalisee.mean(axis=1)
    lambda_max = float(sommes @ poids)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI[n]
    cr = ci / ri if ri > 0 else 0.0
    return {
        "sommes": sommes,
        "normalisee": normalisee,
        "poids": poids,
        "lambda_max": lambda_max,
        "ci": ci,
        "ri": ri,
        "cr": cr,
        "consistant": cr < 0.1,
    }


def ahp_vecteur_propre_exact(A):
    """Vecteur propre principal exact, pour comparer avec la méthode approximative."""
    A = verifier_matrice_comparaison(A)
    valeurs, vecteurs = np.linalg.eig(A)
    k = int(np.argmax(valeurs.real))
    v = np.abs(vecteurs[:, k].real)
    return v / v.sum(), float(valeurs[k].real)


def ahp_complet(A_criteres, matrices_alternatives):
    """Priorités locales puis globales des alternatives (diapos 30-31)."""
    criteres = ahp(A_criteres)
    locales = [ahp(M) for M in matrices_alternatives]
    if len(locales) != len(criteres["poids"]):
        raise ValueError("Il faut une matrice d'alternatives par critère.")
    priorites_locales = np.column_stack([r["poids"] for r in locales])
    return {
        "criteres": criteres,
        "locales": locales,
        "priorites_locales": priorites_locales,
        "priorites_finales": priorites_locales @ criteres["poids"],
    }


def bwm(best_to_others, others_to_worst):
    """Modèle linéaire de Rezaei (diapos 38-43) : min ξ sous |wB - aBj wj| ≤ ξ et |wj - ajW wW| ≤ ξ."""
    bo = np.asarray(best_to_others, dtype=float)
    ow = np.asarray(others_to_worst, dtype=float)
    if bo.shape != ow.shape or bo.ndim != 1:
        raise ValueError("Les vecteurs BO et OW doivent avoir la même longueur.")
    if ((bo < 1) | (bo > 9) | (ow < 1) | (ow > 9)).any():
        raise ValueError("Les notes BO et OW doivent être comprises entre 1 et 9.")
    n = bo.size
    meilleur = int(np.argmin(bo))
    pire = int(np.argmin(ow))
    if meilleur == pire:
        raise ValueError("Le meilleur et le pire critère doivent être différents.")
    if bo[meilleur] != 1 or ow[pire] != 1:
        raise ValueError("Il faut aBB = 1 et aWW = 1.")

    A_ub, lignes = [], []
    for j in range(n):
        if j != meilleur:
            for signe in (1, -1):
                ligne = np.zeros(n + 1)
                ligne[meilleur] += signe
                ligne[j] -= signe * bo[j]
                ligne[-1] = -1
                A_ub.append(ligne)
        if j != pire:
            for signe in (1, -1):
                ligne = np.zeros(n + 1)
                ligne[j] += signe
                ligne[pire] -= signe * ow[j]
                ligne[-1] = -1
                A_ub.append(ligne)
    A_ub = np.array(A_ub)
    c = np.zeros(n + 1)
    c[-1] = 1
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=np.zeros(len(A_ub)),
        A_eq=[[1.0] * n + [0.0]],
        b_eq=[1.0],
        bounds=[(0, None)] * (n + 1),
        method="highs",
    )
    if not res.success:
        raise ValueError(f"Le programme linéaire n'a pas de solution : {res.message}")
    return {
        "poids": res.x[:n],
        "xi": float(res.x[-1]),
        "meilleur": meilleur,
        "pire": pire,
        "nb_comparaisons": 2 * n - 3,
        "nb_comparaisons_ahp": n * (n - 1) // 2,
        "A_ub": A_ub,
        "note_max_sur_pire": int(np.argmax(bo)) == pire,
    }


def _normalisation_lineaire_positive(X, sens):
    """r = x / max (bénéfice), r = min / x (coût) si tout est > 0 ; sinon min-max."""
    X = np.asarray(X, dtype=float)
    R = np.zeros_like(X)
    for j, s in enumerate(sens):
        col = X[:, j]
        if (col > 0).all():
            R[:, j] = col / col.max() if s > 0 else col.min() / col
        else:
            etendue = col.max() - col.min()
            if etendue > 0:
                R[:, j] = (col - col.min()) / etendue if s > 0 else (col.max() - col) / etendue
    return R


def entropie(X, sens):
    """Poids par l'entropie (diapos 44-45), appliquée à la matrice normalisée."""
    R = _normalisation_lineaire_positive(X, sens)
    m, n = R.shape
    if m < 2:
        raise ValueError("L'entropie demande au moins deux alternatives.")
    sommes = R.sum(axis=0)
    p = np.divide(R, sommes, out=np.full_like(R, 1 / m), where=sommes != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        termes = np.where(p > 0, p * np.log(p), 0.0)
    e = -termes.sum(axis=0) / np.log(m)
    diversification = 1 - e
    total = diversification.sum()
    poids = diversification / total if total > 1e-12 else np.full(n, 1 / n)
    return {"normalisee": R, "p": p, "entropie": e, "diversification": diversification, "poids": poids}


def normalisation_min_max(X, sens):
    """Normalisation de CRITIC (diapo 47) : 0 = pire valeur, 1 = meilleure valeur."""
    X = np.asarray(X, dtype=float)
    R = np.zeros_like(X)
    for j, s in enumerate(sens):
        col = X[:, j]
        etendue = col.max() - col.min()
        if etendue > 0:
            R[:, j] = (col - col.min()) / etendue if s > 0 else (col.max() - col) / etendue
    return R


def critic(X, sens):
    """Poids CRITIC (diapos 47-49) : Cj = σj Σ(1 - |ρjk|), wj = Cj / ΣCj."""
    R = normalisation_min_max(X, sens)
    m, n = R.shape
    if m < 3:
        raise ValueError("CRITIC demande au moins trois alternatives pour estimer les corrélations.")
    ecart_type = R.std(axis=0, ddof=1)
    rho = np.eye(n)
    for j in range(n):
        for k in range(j + 1, n):
            if ecart_type[j] > 0 and ecart_type[k] > 0:
                rho[j, k] = rho[k, j] = np.corrcoef(R[:, j], R[:, k])[0, 1]
    independance = (1 - np.abs(rho)).sum(axis=1)
    c = ecart_type * independance
    poids = c / c.sum() if c.sum() > 0 else np.full(n, 1 / n)
    return {"normalisee": R, "ecart_type": ecart_type, "correlation": rho, "indice_c": c, "poids": poids}


def poids_hybrides(subjectifs, objectifs, alpha=0.5):
    """N.B. de la diapo 22 : combinaison convexe expertise / données."""
    w = alpha * np.asarray(subjectifs, dtype=float) + (1 - alpha) * np.asarray(objectifs, dtype=float)
    return w / w.sum()
