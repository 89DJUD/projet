"""Vérifie le moteur contre les résultats publiés dans le cours et le notebook Fuzzy_flou.ipynb."""
import numpy as np
import pytest

from mcdm import alternative_unique as au
from mcdm import classement as cl
from mcdm import exemples as ex
from mcdm import flou, mcdm_flou as mf, ponderation as po, robustesse as ro


def test_ahp_voiture_exercice():
    res = po.ahp(ex.AHP_MATRICE_CRITERES)
    assert res["poids"] == pytest.approx([0.669, 0.088, 0.243], abs=1e-3)
    assert res["cr"] < 0.1
    global_ = po.ahp_complet(ex.AHP_MATRICE_CRITERES, list(ex.AHP_MATRICES_ALTERNATIVES.values()))
    assert global_["priorites_finales"].sum() == pytest.approx(1)
    assert global_["priorites_finales"][0] > global_["priorites_finales"][1]


def test_ahp_refuse_matrice_non_reciproque():
    with pytest.raises(ValueError):
        po.ahp([[1, 2], [2, 1]])


def test_bwm_comparaisons_et_contraintes():
    res = po.bwm(ex.BWM_BO, ex.BWM_OW)
    assert res["poids"].sum() == pytest.approx(1)
    assert res["meilleur"] == 0 and res["pire"] == 3
    assert res["nb_comparaisons"] == 5 and res["nb_comparaisons_ahp"] == 6
    assert res["poids"].argmax() == 0 and res["poids"].argmin() == 3
    # Optimum : toutes les contraintes respectées avec l'écart ξ trouvé
    w, xi = res["poids"], res["xi"]
    assert all(abs(w[0] - a * w[j]) <= xi + 1e-9 for j, a in enumerate(ex.BWM_BO))
    assert all(abs(w[j] - a * w[3]) <= xi + 1e-9 for j, a in enumerate(ex.BWM_OW))


def test_bwm_ecart_avec_les_poids_de_la_diapo_42():
    # La diapo 42 annonce ξ* = 0,08 avec (0,552 ; 0,207 ; 0,155 ; 0,086). Le ξ concorde avec le
    # modèle linéaire, mais ces poids violent ses contraintes : l'optimum trouvé est différent.
    res = po.bwm(ex.BWM_BO, ex.BWM_OW)
    assert res["xi"] == pytest.approx(0.08, abs=0.005)
    w = np.array([0.552, 0.207, 0.155, 0.086])
    ecart_diapo = max(
        max(abs(w[0] - a * w[j]) for j, a in enumerate(ex.BWM_BO)),
        max(abs(w[j] - a * w[3]) for j, a in enumerate(ex.BWM_OW)),
    )
    assert ecart_diapo > 0.2 > res["xi"]


def test_entropie_critere_uniforme_sans_poids():
    X = np.array([[5.0, 1], [5.0, 9], [5.0, 3]])
    res = po.entropie(X, [1, 1])
    assert res["poids"] == pytest.approx([0, 1])


def test_critic_exercice():
    pb = ex.MACHINES_CRITIC
    res = po.critic(pb.X, pb.sens)
    assert res["poids"].sum() == pytest.approx(1)
    assert res["normalisee"][0, 0] == pytest.approx(1)
    assert (res["poids"] > 0).all()


def test_wsm_wpm_waspas_exercice():
    pb = ex.MACHINES_WASPAS
    res = cl.waspas(pb.X, pb.poids, pb.sens)
    assert res["normalisee"][1, 0] == pytest.approx(1)
    assert res["normalisee"][0, 0] == pytest.approx(0.027 / 0.035)
    assert res["scores"] == pytest.approx((res["q1"] + res["q2"]) / 2)
    assert res["rangs"].min() == 1


def test_topsis_exercice():
    pb = ex.VOITURES_TOPSIS
    res = cl.topsis(pb.X, pb.poids, pb.sens)
    assert ((0 <= res["scores"]) & (res["scores"] <= 1)).all()
    # Calcul manuel de la première cellule normalisée
    assert res["normalisee"][0, 0] == pytest.approx(7 / np.sqrt(49 + 64 + 81 + 36))


def test_vikor_exercice():
    pb = ex.MACHINES_VIKOR
    res = cl.vikor(pb.X, pb.poids, pb.sens)
    assert res["S"] == pytest.approx([0.5, 0.375, 0.5])
    assert res["R"] == pytest.approx([0.5, 0.25, 0.5])
    assert res["Q"] == pytest.approx([1, 0, 1])
    assert res["compromis"] == [1]
    assert res["avantage_acceptable"] and res["stabilite_acceptable"]


def test_methodes_sur_matrice_a_valeurs_negatives():
    pb = ex.ACTIONS_VUKOVIC
    assert sum(pb.poids) == pytest.approx(1, abs=1e-3)  # arrondis de la diapo 18
    rangs, _, erreurs = cl.classer_toutes_methodes(pb.X, pb.poids, pb.sens)
    assert "WPM" in erreurs and "WASPAS" in erreurs
    assert {"WSM", "TOPSIS", "VIKOR"} <= set(rangs)


def test_spearman_diapo_79():
    r = ex.SPEARMAN_RANGS
    assert ro.spearman(r["TOPSIS"], r["VIKOR"]) == pytest.approx(0.77, abs=0.005)
    assert ro.rho_critique(6) == 0.886


def test_maut_utilites():
    res = au.maut([5, 5], [0.5, 0.5], [1, -1], [0, 0], [10, 10])
    assert res["utilites"] == pytest.approx([0.5, 0.5])
    assert res["score"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "seuil,tpr,fpr,d,j",
    [(0.75, 0.67, 0.00, 0.33, 0.67), (0.65, 1.00, 0.33, 0.33, 0.67), (0.60, 1.00, 0.67, 0.67, 0.33)],
)
def test_seuil_diapo_88(seuil, tpr, fpr, d, j):
    m = au.metriques_seuil(ex.SEUIL_SCORES, ex.SEUIL_DECISIONS, seuil)
    assert m["TPR"] == pytest.approx(tpr, abs=0.005)
    assert m["FPR"] == pytest.approx(fpr, abs=0.005)
    assert m["distance"] == pytest.approx(d, abs=0.005)
    assert m["youden"] == pytest.approx(j, abs=0.005)


def test_seuil_recherche_trouve_separation_parfaite():
    lignes = au.table_roc(ex.SEUIL_SCORES, ex.SEUIL_DECISIONS)
    meilleur = au.meilleurs_seuils(lignes)["Indice de Youden"]
    assert meilleur["youden"] == pytest.approx(1)
    assert 0.69 < meilleur["seuil"] <= 0.71
    assert au.auc(lignes) == pytest.approx(1)


def test_intuitionniste_diapos_114_117():
    assert flou.hesitation_ifn((0.6, 0.3)) == pytest.approx(0.1)
    assert flou.ifn_somme((0.5, 0.3), (0.5, 0.3)) == pytest.approx((0.75, 0.09))
    assert flou.ifwa([(0.5, 0.3), (0.5, 0.3)], [0.5, 0.5]) == pytest.approx((0.5, 0.3))
    assert flou.ifn_comparer((0.6, 0.3), (0.5, 0.2)) == 1  # même score, meilleure précision


def test_score_flou_diapo_134():
    res = mf.score_flou([(1, 1.2, 1.5), (110, 120, 130)], [0.6, 0.4], [1.5, 130])
    assert res["agrege"] == pytest.approx([0.74, 0.85, 1.0], abs=0.005)
    assert res["score"] == pytest.approx(0.86, abs=0.005)
    assert res["incertitude"] == pytest.approx(0.26, abs=0.005)


def test_topsis_flou_notebook():
    notes = mf.notes_vers_tableau(ex.RECRUTEMENT_NOTES, mf.ECHELLE_NOTES)
    poids = mf.poids_vers_tableau(ex.RECRUTEMENT_POIDS, mf.ECHELLE_POIDS)
    res = mf.topsis_flou(notes, poids)
    assert res["scores"] == pytest.approx([0.6380, 0.7653, 0.7018], abs=1e-4)
    assert res["d_plus"] == pytest.approx([1.9456, 1.2589, 1.6020], abs=1e-4)
    assert list(res["rangs"]) == [3, 1, 2]


def test_topsis_flou_cout_triplet_croissant():
    notes = mf.notes_vers_tableau(ex.RECRUTEMENT_NOTES, mf.ECHELLE_NOTES)
    poids = mf.poids_vers_tableau(ex.RECRUTEMENT_POIDS, mf.ECHELLE_POIDS)
    res = mf.topsis_flou(notes, poids, sens=[-1, 1, 1, 1, 1])
    r = res["normalisee"][:, 0]
    assert (r[:, 0] <= r[:, 1]).all() and (r[:, 1] <= r[:, 2]).all()


def test_topsis_intuitionniste_recrutement():
    notes = mf.notes_vers_tableau(ex.RECRUTEMENT_NOTES, mf.ECHELLE_NOTES_IF)
    poids = mf.poids_vers_tableau(ex.RECRUTEMENT_POIDS, mf.ECHELLE_POIDS_IF)
    res = mf.topsis_intuitionniste(notes, poids)
    assert ((0 <= res["scores"]) & (res["scores"] <= 1)).all()
    assert res["rangs"][1] == 1  # A2 domine aussi avec des nombres intuitionnistes
    # Un décideur unique rend l'agrégation neutre
    seul = mf.topsis_intuitionniste(notes[:, :1], poids[:1])
    assert seul["agregee"][0, 0] == pytest.approx(mf.ECHELLE_NOTES_IF["MG"])
