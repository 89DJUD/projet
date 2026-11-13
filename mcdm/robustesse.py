"""Comparaison des classements (Spearman) et analyse de sensibilité aux poids."""
import numpy as np
from scipy.stats import rankdata


def spearman(rangs_1, rangs_2):
    """ρ de Spearman (diapo 77). Avec des ex æquo, la formule en d² n'est plus exacte :
    on utilise alors la corrélation de Pearson sur les rangs moyens, sa généralisation."""
    r1, r2 = np.asarray(rangs_1, dtype=float), np.asarray(rangs_2, dtype=float)
    n = r1.size
    if n < 2:
        return 1.0
    if np.unique(r1).size == n and np.unique(r2).size == n:
        return float(1 - 6 * ((r1 - r2) ** 2).sum() / (n * (n**2 - 1)))
    a, b = rankdata(r1), rankdata(r2)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# Valeurs critiques de ρ à 5 % en bilatéral (table de Spearman). La diapo 79 donne 0,886 pour n = 6.
RHO_CRITIQUE = {5: 1.0, 6: 0.886, 7: 0.786, 8: 0.738, 9: 0.700, 10: 0.648, 12: 0.587, 14: 0.538, 16: 0.503, 18: 0.472, 20: 0.447}


def rho_critique(n):
    """Valeur critique tabulée, ou celle de la taille tabulée immédiatement inférieure (plus prudente)."""
    tailles = [t for t in sorted(RHO_CRITIQUE) if t <= n]
    return RHO_CRITIQUE[tailles[-1]] if tailles else None


def matrice_spearman(rangs):
    noms = list(rangs)
    M = np.eye(len(noms))
    for i, a in enumerate(noms):
        for j, b in enumerate(noms):
            if j > i:
                M[i, j] = M[j, i] = spearman(rangs[a], rangs[b])
    return noms, M


def balayage_poids(fonction_classement, X, poids, sens, critere, pas=21):
    """Fait varier le poids d'un critère de 0 à 1, les autres restant proportionnels."""
    poids = np.asarray(poids, dtype=float)
    valeurs = np.linspace(0, 1, pas)
    autres = np.delete(poids, critere)
    total = autres.sum()
    resultats = []
    for w in valeurs:
        if total > 0:
            nouveau = poids * (1 - w) / total
        else:
            nouveau = np.full_like(poids, (1 - w) / max(poids.size - 1, 1))
        nouveau[critere] = w
        res = fonction_classement(X, nouveau, sens)
        resultats.append((w, res["scores"], res["rangs"]))
    return resultats


def seuil_inversion(fonction_classement, X, poids, sens, pas=101):
    """Pour chaque critère, le poids le plus proche de l'actuel où la meilleure alternative change."""
    poids = np.asarray(poids, dtype=float)
    meilleure = int(np.argmin(fonction_classement(X, poids, sens)["rangs"]))
    sortie = []
    for j in range(poids.size):
        candidats = []
        for w, _, rangs in balayage_poids(fonction_classement, X, poids, sens, j, pas=pas):
            if int(np.argmin(rangs)) != meilleure:
                candidats.append(w)
        sortie.append(min(candidats, key=lambda w: abs(w - poids[j])) if candidats else None)
    return meilleure, sortie


def agreger_rangs(rangs):
    """Consensus de Borda : moyenne des rangs à travers les méthodes."""
    noms = list(rangs)
    if not noms:
        raise ValueError("Aucune méthode n'a produit de classement.")
    M = np.column_stack([rangs[n] for n in noms])
    moyenne = M.mean(axis=1)
    return {
        "rang_moyen": moyenne,
        "rang_consensus": rankdata(moyenne, method="min").astype(int),
        "nb_premier": (M == 1).sum(axis=1),
        "nb_methodes": len(noms),
    }
