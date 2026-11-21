"""Classification par règles des pièces détectées (approche classique, sans IA).

Décision en trois temps, pour toutes les pièces d'une même image :

1. **Métal** (couleur Lab, voir features.py)
   - bimétal (5 dinars) si b*(anneau) - b*(centre) >= `bimetal_min_contrast` ;
   - sinon doré ou argenté, par ordre de priorité :
     a. si la photo contient un 5 dinars : on le prend comme référence (son centre est
        argenté, son anneau doré, sous le même éclairage) et on coupe au milieu ;
     b. si les b* corrigés du fond (b - alpha * b_fond) forment deux groupes nettement
        séparés, on coupe entre ces deux groupes (comparaison relative) ;
     c. sinon seuil fixe `gold_threshold`.

2. **Forme** : un score de 13e harmonique >= `polygon_min_score` indique une pièce à
   13 côtés (200 millimes si dorée, 2 dinars si argentée). L'indice n'est utilisé que
   dans un sens : un score faible n'exclut pas ces deux classes (rappel limité).

3. **Taille relative** : la taille en pixels dépend de la distance, mais toutes les pièces
   d'une photo partagent la même échelle s (pixels par mm). On cherche l'échelle s qui
   explique le mieux tous les rayons de l'image par les diamètres réels des classes
   compatibles avec le métal et la forme de chaque pièce :

       coût(s) = Σ_i min_c min(e_ic², plafond²) + poids_a_priori · log(s / s0)²
       avec e_ic = log( (2 r_i / s) / diamètre_c )

   Le plafond limite l'effet d'une pièce mal détectée. Le terme a priori (faible) ne sert
   qu'à départager des échelles équivalentes ; s0 est mesuré sur les photos L1.

**Confiance** : pour chaque pièce, on calcule de combien le coût minimal augmente si on
lui interdit sa classe retenue. Une petite augmentation signifie qu'une autre lecture de
l'image est presque aussi plausible (ex. une pièce dorée seule : sa taille ne prouve rien).
confiance = 1 - exp(-augmentation / tau). En dessous de `min_confidence`, ou si l'écart au
diamètre attendu dépasse `max_relative_error`, la pièce est déclarée « inconnue ».
Sur les images de développement, rejeter selon la confiance faisait perdre plus de
pièces correctes qu'il n'évitait d'erreurs : `min_confidence` vaut donc 0 et la confiance
est seulement affichée (voir configs/classical.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.classical.detection import CoinCandidate
from src.classical.features import CoinFeatures
from src.common.coin_classes import (COIN_DIAMETERS_MM, COIN_METAL, POLYGONAL_CLASSES,
                                     UNKNOWN_LABEL)


@dataclass
class CoinPrediction:
    """Résultat de classification d'une pièce."""

    coin: CoinCandidate
    label: str            # classe retenue ou UNKNOWN_LABEL
    best_guess: str       # meilleure classe même si la confiance est insuffisante
    confidence: float     # 0 à 1
    metal: str            # gold | silver | bimetal
    polygonal: bool
    features: CoinFeatures


def assign_metals(feats: list[CoinFeatures], cfg: dict) -> list[str]:
    """Métal de chaque pièce : 'bimetal', 'gold' ou 'silver'."""
    metals = ["bimetal" if f.bimetal_contrast >= cfg["bimetal_min_contrast"] else ""
              for f in feats]
    plain = [i for i, m in enumerate(metals) if not m]
    if not plain:
        return metals

    # Référence interne : le 5 dinars a un centre argenté et un anneau doré, vus sous
    # le même éclairage que les autres pièces de la photo.
    refs = [f for f, m in zip(feats, metals) if m == "bimetal"]
    if refs:
        threshold = np.mean([(f.b_center + f.b_ring) / 2 for f in refs]) + cfg["reference_offset"]
        for i in plain:
            metals[i] = "gold" if feats[i].b > threshold else "silver"
        return metals

    values = np.array([feats[i].corrected_b(cfg["background_alpha"]) for i in plain])
    threshold = cfg["gold_threshold"]

    if len(plain) >= 2:
        order = np.argsort(values)
        sorted_v = values[order]
        k = int(np.diff(sorted_v).argmax())          # plus grand écart entre deux valeurs
        low, high = sorted_v[:k + 1].mean(), sorted_v[k + 1:].mean()
        # Deux groupes nettement séparés, dont le plus bas est plausiblement argenté
        if high - low >= cfg["split_min_gap"] and low < threshold:
            threshold = (sorted_v[k] + sorted_v[k + 1]) / 2

    for i, v in zip(plain, values):
        metals[i] = "gold" if v > threshold else "silver"
    return metals


def _candidates(metal: str, polygonal: bool) -> list[str]:
    classes = [c for c, m in COIN_METAL.items() if m == metal]
    if polygonal:
        poly = [c for c in classes if c in POLYGONAL_CLASSES]
        if poly:
            return poly
    return classes


def classify_coins(coins: list[CoinCandidate], feats: list[CoinFeatures],
                   cfg: dict, min_side: int) -> list[CoinPrediction]:
    """Classe toutes les pièces d'une image (rayons en pixels de l'image traitée)."""
    if not coins:
        return []
    metals = assign_metals(feats, cfg)
    polys = [f.poly_score >= cfg["polygon_min_score"] for f in feats]
    cands = [_candidates(m, p) for m, p in zip(metals, polys)]
    radii = np.array([f.r_edge for f in feats])

    # Grille d'échelles s (pixels par mm), assez large pour toutes les lectures possibles
    d_all = np.array(list(COIN_DIAMETERS_MM.values()))
    s_grid = np.geomspace(0.7 * 2 * radii.min() / d_all.max(),
                          1.4 * 2 * radii.max() / d_all.min(), cfg["scale_steps"])
    cap2 = cfg["error_cap"] ** 2
    s0 = cfg["scale_prior_px_per_mm_per_side"] * min_side
    prior = cfg["scale_prior_weight"] * np.log(s_grid / s0) ** 2

    # err[i][s, j] : erreur log du diamètre si la pièce i est de la classe cands[i][j]
    err = [np.log((2 * r / s_grid)[:, None] / np.array([COIN_DIAMETERS_MM[c] for c in cs]))
           for r, cs in zip(radii, cands)]
    sq = [np.minimum(e ** 2, cap2) for e in err]
    terms = np.stack([q.min(axis=1) for q in sq], axis=1)        # (n_s, n_coins)
    total = terms.sum(axis=1) + prior
    s_best = int(total.argmin())
    best_cost = total[s_best]

    preds = []
    for i, coin in enumerate(coins):
        j = int(sq[i][s_best].argmin())
        guess = cands[i][j]
        rel_err = float(abs(err[i][s_best, j]))
        if len(cands[i]) == 1:
            # Classe imposée par la couleur ou la forme : un rival hors candidats
            # coûterait le plafond (aucune autre classe compatible).
            alt_term = np.full(len(s_grid), cap2)
        else:
            alt_term = np.delete(sq[i], j, axis=1).min(axis=1)
        alt_cost = (total - terms[:, i] + alt_term).min()
        conf = float(1 - np.exp(-(alt_cost - best_cost) / cfg["confidence_tau"]))
        ok = conf >= cfg["min_confidence"] and rel_err <= cfg["max_relative_error"]
        preds.append(CoinPrediction(coin=coin, label=guess if ok else UNKNOWN_LABEL,
                                    best_guess=guess, confidence=conf, metal=metals[i],
                                    polygonal=polys[i], features=feats[i]))
    return preds
