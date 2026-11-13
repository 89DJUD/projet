"""Évaluation absolue d'une seule alternative : MAUT, choix du seuil (ROC), distance à l'idéal."""
import numpy as np


def maut(valeurs, poids, sens, mins, maxs):
    """S(A) = Σ wj uj(xj) avec une utilité linéaire bornée à [0, 1] (diapo 82)."""
    x, w = np.asarray(valeurs, dtype=float), np.asarray(poids, dtype=float)
    mins, maxs = np.asarray(mins, dtype=float), np.asarray(maxs, dtype=float)
    if (maxs <= mins).any():
        raise ValueError("Pour chaque critère, la borne max doit dépasser la borne min.")
    u = np.where(np.asarray(sens) > 0, (x - mins) / (maxs - mins), (x - maxs) / (mins - maxs))
    u = np.clip(u, 0, 1)
    return {"utilites": u, "score": float(u @ w)}


def distance_ideal(valeurs, poids, sens, mins, maxs):
    """Indice de performance façon TOPSIS pour une seule alternative (diapo 89)."""
    x, w = np.asarray(valeurs, dtype=float), np.asarray(poids, dtype=float)
    sens = np.asarray(sens)
    etendue = np.asarray(maxs, dtype=float) - np.asarray(mins, dtype=float)
    if (etendue <= 0).any():
        raise ValueError("Pour chaque critère, la borne max doit dépasser la borne min.")
    # Unités différentes : on met chaque critère à l'échelle de son étendue avant les distances.
    ideal = np.where(sens > 0, maxs, mins)
    anti = np.where(sens > 0, mins, maxs)
    d_plus = float(np.sqrt((w * ((x - ideal) / etendue) ** 2).sum()))
    d_moins = float(np.sqrt((w * ((x - anti) / etendue) ** 2).sum()))
    total = d_plus + d_moins
    return {"d_plus": d_plus, "d_moins": d_moins, "indice": d_moins / total if total else 0.0}


def metriques_seuil(scores, decisions, seuil):
    """Matrice de confusion et indicateurs de la diapo 87-88 pour un seuil τ."""
    s, y = np.asarray(scores, dtype=float), np.asarray(decisions, dtype=int)
    pred = s >= seuil
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "seuil": float(seuil),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "TPR": tpr,
        "FPR": fpr,
        "distance": float(np.sqrt(fpr**2 + (1 - tpr) ** 2)),
        "youden": tpr - fpr,
        "exactitude": (tp + tn) / max(s.size, 1),
    }


def seuils_candidats(scores):
    """Un seuil au-dessus de tout, puis le milieu entre deux scores consécutifs, puis sous tout."""
    uniques = np.unique(np.asarray(scores, dtype=float))
    milieux = (uniques[:-1] + uniques[1:]) / 2
    return np.concatenate(([uniques[-1] + 0.01], milieux[::-1], [uniques[0] - 0.01]))


def table_roc(scores, decisions, cout_fp=1.0, cout_fn=1.0):
    y = np.asarray(decisions, dtype=int)
    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("Il faut au moins une décision « accepter » et une décision « rejeter ».")
    lignes = []
    for t in seuils_candidats(scores):
        m = metriques_seuil(scores, y, t)
        m["cout"] = cout_fp * m["FP"] + cout_fn * m["FN"]
        lignes.append(m)
    return lignes


def auc(lignes):
    fpr = np.array([l["FPR"] for l in lignes])
    tpr = np.array([l["TPR"] for l in lignes])
    ordre = np.lexsort((tpr, fpr))
    return float(np.trapezoid(tpr[ordre], fpr[ordre]))


def meilleurs_seuils(lignes):
    return {
        "Distance au point idéal": min(lignes, key=lambda l: l["distance"]),
        "Indice de Youden": max(lignes, key=lambda l: l["youden"]),
        "Exactitude": max(lignes, key=lambda l: l["exactitude"]),
        "Coût minimal": min(lignes, key=lambda l: l["cout"]),
    }
