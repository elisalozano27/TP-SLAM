# localisation_grid.py - Localisation d'un scan sur une grid map par scan matching
# Suit le document « Scan matching : méthode de localisation en 2D avec Lidar »
#
# Programme principal : localise le robot sur TA grid map avec les 2 meilleures méthodes
# trouvées par localisation_site.py :
#   - multi-résolution : sans estimation initiale (démarrage, robot perdu)
#   - descente de gradient : à partir d'une estimation initiale (suivi du robot)
# La grid map est construite avec tous les scans de poses.csv SAUF le scan à localiser.
# Si ce scan est dans poses.csv, on compare le résultat à sa pose (alignée à la main).
#
# Les 4 étapes du scan matching (§3 du document) :
#   1. Transformation : appliquer une pose candidate (x, y, θ) aux points du scan
#   2. Corrélation    : projeter les points transformés dans la grid map
#   3. Score          : mesurer la qualité de l'alignement (3 scores possibles, §4)
#   4. Optimisation   : chercher la pose qui maximise le score (4 méthodes, §8)

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from grille import lire_scan, charger_poses, nouvelle_carte, mettre_a_jour, INCONNU, RESOLUTION, N, TAILLE

SCAN_TEST = "scan_05.csv"           # scan à localiser (pas utilisé pour construire la carte)
NB_POINTS = 400                     # prétraitement : sous-échantillonnage du scan (§3.2)
ERREUR_DEPART = [0.40, -0.30, 8.0]  # scan de poses.csv : estimation initiale = sa pose + cette erreur
ESTIMATION_INITIALE = None          # nouveau scan (absent de poses.csv) : pose approximative [x, y, θ]
                                    # pour le gradient, ou None pour n'utiliser que la multi-résolution

# Score (§4) : "occupees"     = proportion de points dans des cellules occupées
#              "probabilites" = moyenne des probabilités d'occupation sous les points
#              "distance"     = proximité des cellules occupées les plus proches
SCORE = "distance"
D_MAX = 0.50                        # score "distance" : au-delà de D_MAX d'un mur, le score vaut 0 (m)

# Convergence (§3.4) : on s'arrête si le score progresse de moins de SEUIL_SCORE, ou après IT_MAX itérations
SEUIL_SCORE = 1e-6
IT_MAX = 200


# ---------- Référence : la grid map préparée pour le score ----------
def distance_aux_murs(occupe, res, d_max):
    """Distance (m) de chaque cellule à la cellule occupée la plus proche, limitée à d_max.
    On propage la distance de voisine en voisine (8 directions), comme une tache d'huile."""
    d = np.where(occupe, 0.0, np.inf)
    voisins = [(dl, dc, res * math.hypot(dl, dc)) for dl in (-1, 0, 1) for dc in (-1, 0, 1) if dl or dc]
    for _ in range(math.ceil(d_max / res) + 1):
        for dl, dc, cout in voisins:
            d = np.minimum(d, np.roll(np.roll(d, dl, 0), dc, 1) + cout)
    return np.minimum(d, d_max)


def reference(proba, res, origine, facteur=1, d_max=D_MAX):
    """Prépare une grid map pour le score choisi.
    proba   : tableau des probabilités d'occupation (0 à 1), ligne = y, colonne = x
    res     : taille d'une cellule (m) ; origine : position (x, y) du coin de la cellule [0, 0]
    facteur > 1 : carte sous-échantillonnée, des cellules facteur fois plus grandes (§8.2)."""
    if facteur > 1:
        nl, nc = -(-proba.shape[0] // facteur), -(-proba.shape[1] // facteur)   # arrondi au-dessus
        proba = np.pad(proba, ((0, nl * facteur - proba.shape[0]), (0, nc * facteur - proba.shape[1])))
        proba = proba.reshape(nl, facteur, nc, facteur).max(axis=(1, 3))
        res = res * facteur
    l, c = np.nonzero(proba > 0.5)                     # centres des cellules occupées (pour le §8.5)
    cellules = np.column_stack([origine[0] + (c + 0.5) * res, origine[1] + (l + 0.5) * res])
    # On ajoute 2 cellules tout autour, copies du bord : un point qui tombe juste sur le bord
    # extérieur d'un mur de la limite de carte n'est pas compté comme « hors carte »
    proba = np.pad(proba, 2, mode="edge")
    origine = (origine[0] - 2 * res, origine[1] - 2 * res)
    occupe = proba > 0.5
    if SCORE == "occupees":
        tab = occupe.astype(float)
    elif SCORE == "probabilites":
        tab = proba * occupe                           # les cellules libres ou inconnues comptent 0
    else:
        tab = 1 - distance_aux_murs(occupe, res, d_max) / d_max   # 1 sur un mur, 0 à d_max et plus
    return {"tab": tab, "res": res, "origine": origine, "cellules": cellules}


def reference_grille(carte, facteur=1, d_max=D_MAX):
    """Référence à partir d'une carte de grille.py (0 à 255, centrée sur (0, 0))."""
    return reference(carte / 255.0, RESOLUTION, (-N / 2 * RESOLUTION, -N / 2 * RESOLUTION), facteur, d_max)


def reference_vectorielle(segments, d_max=D_MAX):
    """Référence à partir d'une carte vectorielle : liste de segments ((x1, y1), (x2, y2))."""
    return {"segments": np.array([[a[0], a[1], b[0], b[1]] for a, b in segments], dtype=float), "d_max": d_max}


def zone_occupee(ref):
    """Rectangle ((xmin, xmax), (ymin, ymax)) qui contient tous les murs de la carte."""
    if "segments" in ref:
        s = ref["segments"]
        return (s[:, [0, 2]].min(), s[:, [0, 2]].max()), (s[:, [1, 3]].min(), s[:, [1, 3]].max())
    lignes, colonnes = np.nonzero(ref["tab"] >= 1)
    (ox, oy), r = ref["origine"], ref["res"]
    return (ox + colonnes.min() * r, ox + (colonnes.max() + 1) * r), (oy + lignes.min() * r, oy + (lignes.max() + 1) * r)


# ---------- Étapes 1, 2 et 3 : transformation, corrélation, score ----------
def transformer(pts, x, y, theta):
    """Étape 1 : points du repère du robot -> repère monde (rotation de θ puis translation).
    x, y, theta peuvent être des tableaux : on transforme alors pour plusieurs poses à la fois."""
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    X = np.asarray(x)[..., None] + np.asarray(c)[..., None] * pts[:, 0] - np.asarray(s)[..., None] * pts[:, 1]
    Y = np.asarray(y)[..., None] + np.asarray(s)[..., None] * pts[:, 0] + np.asarray(c)[..., None] * pts[:, 1]
    return X, Y


def correler(ref, X, Y):
    """Étape 2 : projection des points dans la grid map. On lit la valeur sous chaque point
    par interpolation bilinéaire (mélange des 4 cellules voisines). 0 hors de la carte."""
    tab, res, (ox, oy) = ref["tab"], ref["res"], ref["origine"]
    nl, nc = tab.shape
    fx, fy = (X - ox) / res - 0.5, (Y - oy) / res - 0.5   # position en cellules (centres des cellules)
    c0, l0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    wx, wy = fx - c0, fy - l0
    dedans = (c0 >= 0) & (c0 < nc - 1) & (l0 >= 0) & (l0 < nl - 1)
    c0, l0 = np.clip(c0, 0, nc - 2), np.clip(l0, 0, nl - 2)
    v = (tab[l0, c0] * (1 - wx) * (1 - wy) + tab[l0, c0 + 1] * wx * (1 - wy)
         + tab[l0 + 1, c0] * (1 - wx) * wy + tab[l0 + 1, c0 + 1] * wx * wy)
    return v * dedans


def segment_le_plus_proche(segments, X, Y):
    """Étape 2 pour une carte vectorielle (§5) : pour chaque point, distance au segment le plus proche
    et point le plus proche sur ce segment (projection du point sur le segment)."""
    d2_min = np.full(np.shape(X), np.inf)
    QX, QY = np.zeros(np.shape(X)), np.zeros(np.shape(X))
    for ax, ay, bx, by in segments:
        ux, uy = bx - ax, by - ay
        t = np.clip(((X - ax) * ux + (Y - ay) * uy) / max(ux * ux + uy * uy, 1e-12), 0, 1)
        qx, qy = ax + t * ux, ay + t * uy                  # point le plus proche sur ce segment
        d2 = (X - qx) ** 2 + (Y - qy) ** 2
        mieux = d2 < d2_min
        d2_min, QX, QY = np.where(mieux, d2, d2_min), np.where(mieux, qx, QX), np.where(mieux, qy, QY)
    return np.sqrt(d2_min), QX, QY


def score(ref, pts, x, y, theta):
    """Étape 3 : score entre 0 et 1, plus haut = mieux aligné.
    Grid map (§4)         : moyenne des valeurs de la carte sous les points.
    Carte vectorielle (§5) : 1 - moyenne des d²/d_max², d = distance au segment le plus proche.
    Maximiser ce score = minimiser E = Σ wᵢ dᵢ² du document, avec wᵢ = 0 pour les points à plus
    de d_max de tout segment (points aberrants ou objets absents de la carte)."""
    X, Y = transformer(pts, x, y, theta)
    if "segments" in ref:
        d = np.minimum(segment_le_plus_proche(ref["segments"], X, Y)[0], ref["d_max"])
        return 1 - (d ** 2).mean(axis=-1) / ref["d_max"] ** 2
    return correler(ref, X, Y).mean(axis=-1)


# ---------- Étape 4 : optimisation (§8) ----------
def force_brute(ref, pts, centre, demi_xy, demi_theta, pas_xy, pas_theta):
    """§8.1 : fenêtre autour de l'estimation initiale, discrétisée, on teste toutes les poses."""
    xs = np.arange(centre[0] - demi_xy, centre[0] + demi_xy + 1e-9, pas_xy)
    ys = np.arange(centre[1] - demi_xy, centre[1] + demi_xy + 1e-9, pas_xy)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    meilleur, meilleure_pose = -1, None
    for theta in np.arange(centre[2] - demi_theta, centre[2] + demi_theta + 1e-9, pas_theta):
        tab = score(ref, pts, X, Y, theta)             # toutes les positions de la fenêtre d'un coup
        i, j = np.unravel_index(tab.argmax(), tab.shape)
        if tab[i, j] > meilleur:
            meilleur, meilleure_pose = tab[i, j], [xs[i], ys[j], theta]
    return meilleure_pose


RAYON = 1.0   # 1 rad de rotation compte comme RAYON m, pour mélanger x, y et θ dans un même pas


def descente_gradient(ref, pts, depart, pas=0.10):
    """§8.3 : 1. approximation du gradient (petites différences), 2. mise à jour de (x, y, θ),
    3. répétition jusqu'à convergence (§3.4). On monte vers le meilleur score ; si un pas
    fait baisser le score, on le divise par 2, sinon on l'agrandit un peu."""
    v = np.array([depart[0], depart[1], math.radians(depart[2]) * RAYON])
    f = lambda v: score(ref, pts, v[0], v[1], math.degrees(v[2] / RAYON))
    actuel, h = f(v), 0.01
    for _ in range(IT_MAX):
        grad = np.array([(f(v + e * h) - f(v - e * h)) / (2 * h) for e in np.eye(3)])
        norme = np.linalg.norm(grad)
        if norme == 0:
            break                                      # carte plate ici : aucune direction à suivre
        essai = v + pas * grad / norme
        nouveau = f(essai)
        if nouveau > actuel:
            progres = nouveau - actuel
            v, actuel, pas = essai, nouveau, pas * 1.2
            if progres < SEUIL_SCORE:
                break                                  # le score ne bouge presque plus : convergé
        else:
            pas /= 2
            if pas < 1e-3:
                break                                  # la pose ne bouge presque plus : convergé
    return [v[0], v[1], math.degrees(v[2] / RAYON)]


NB_CANDIDATS = 5   # multi-résolution : nb de poses gardées à basse résolution


def multi_resolution(ref_basse, ref_haute, pts, zone):
    """§8.2 : 1. downsampling de la carte (cellules de 20 cm) et du scan (1 point sur 4),
    2. recherche brute à basse résolution dans toute la zone, sans estimation initiale,
    3. raffinement progressif à haute résolution (force brute, puis gradient)."""
    (xmin, xmax), (ymin, ymax) = zone
    xs, ys = np.arange(xmin, xmax, 0.20), np.arange(ymin, ymax, 0.20)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    candidats = []
    for theta in np.arange(-180, 180, 4):
        tab = score(ref_basse, pts[::4], X, Y, theta)
        i, j = np.unravel_index(tab.argmax(), tab.shape)
        candidats.append((tab[i, j], [xs[i], ys[j], theta]))
    candidats.sort(key=lambda c: c[0], reverse=True)
    gardes = []                                        # on garde plusieurs poses bien différentes :
    for _, p in candidats:                             # la bonne n'est pas toujours la 1re à basse résolution
        if all(math.dist(p[:2], q[:2]) > 1.0 or erreur(p, q)[1] > 20 for q in gardes):
            gardes.append(p)
        if len(gardes) == NB_CANDIDATS:
            break
    resultats = []
    for p in gardes:
        p = force_brute(ref_haute, pts, p, 0.30, 6, 0.05, 1)   # haute résolution : ±30 cm, ±6°
        p = descente_gradient(ref_haute, pts, p, pas=0.02)      # raffinement final par gradient
        resultats.append((score(ref_haute, pts, *p), p))
    return max(resultats, key=lambda r: r[0])[1]


NB_PARTICULES = 3000
ITERATIONS_PF = 25


def filtre_particules(ref_basse, ref_haute, pts, zone, graine=0):
    """§8.4 : on explore un ensemble d'hypothèses de pose (les particules) au lieu d'une seule.
    Chaque particule reçoit un poids selon son score ; on tire les nouvelles particules surtout
    parmi les plus probables, avec un peu de hasard (bruit) pour ne pas rester dans un minimum local.
    Le bruit diminue à chaque itération : les particules se resserrent autour de la bonne pose."""
    rng = np.random.default_rng(graine)
    (xmin, xmax), (ymin, ymax) = zone
    P = np.column_stack([rng.uniform(xmin, xmax, NB_PARTICULES), rng.uniform(ymin, ymax, NB_PARTICULES),
                         rng.uniform(-180, 180, NB_PARTICULES)])
    bruit = np.array([0.30, 0.30, 10.0])               # bruit de départ : 30 cm, 10°
    for it in range(ITERATIONS_PF):
        ref, p = (ref_basse, pts[::4]) if it < ITERATIONS_PF // 2 else (ref_haute, pts)
        s = score(ref, p, P[:, 0], P[:, 1], P[:, 2])
        poids = np.exp((s - s.max()) / 0.02)           # plus le score est haut, plus le poids est fort
        poids /= poids.sum()
        P = P[rng.choice(len(P), len(P), p=poids)]     # rééchantillonnage selon les poids
        P = P + rng.normal(0, 1, P.shape) * bruit      # un peu de hasard autour de chaque particule
        bruit = np.maximum(bruit * 0.8, [0.01, 0.01, 0.3])
    s = score(ref_haute, pts, P[:, 0], P[:, 1], P[:, 2])
    return list(P[s.argmax()])


# ---------- §8.5 : calcul direct (moments géométriques) et sa version répétée (ICP) ----------
REJET = 0.30          # correspondance ignorée si le point est à plus de REJET de sa cellule (m) : outliers


def un_calcul_direct(ref, pts, pose):
    """Une étape de calcul direct, sans recherche :
    1. correspondances : chaque point du scan est associé à la cellule occupée la plus proche
       (grid map) ou au point le plus proche du segment le plus proche (carte vectorielle) ;
    2. moments : barycentres des deux nuages et covariance entre eux ;
    3. transformation : la rotation et la translation qui superposent au mieux les deux nuages
       s'obtiennent directement par une formule (moindres carrés)."""
    X, Y = transformer(pts, *pose)
    P = np.column_stack([X, Y])                                    # points du scan dans le monde
    if "segments" in ref:                                          # carte vectorielle : point le plus
        dist, qx, qy = segment_le_plus_proche(ref["segments"], X, Y)   # proche sur le segment le plus proche
        Q = np.column_stack([qx, qy])
    else:                                                          # grid map : cellule occupée la plus proche
        proche, dist = [], []
        for k in range(0, len(P), 50):                             # par paquets de 50 points (mémoire)
            d2 = ((P[k:k + 50, None, :] - ref["cellules"][None, :, :]) ** 2).sum(axis=2)
            proche.append(d2.argmin(axis=1))
            dist.append(np.sqrt(d2.min(axis=1)))
        Q, dist = ref["cellules"][np.concatenate(proche)], np.concatenate(dist)
    bons = dist < REJET                                            # on écarte les correspondances aberrantes
    if bons.sum() < 3:
        return pose
    P, Q = P[bons], Q[bons]
    mp, mq = P.mean(axis=0), Q.mean(axis=0)                        # barycentres (moments d'ordre 1)
    p, q = P - mp, Q - mq
    dtheta = math.atan2((p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum(),   # rotation optimale
                        (p[:, 0] * q[:, 0] + p[:, 1] * q[:, 1]).sum())   # (moments d'ordre 2)
    c, s = math.cos(dtheta), math.sin(dtheta)
    # nouveau point = R(dθ) · (point - mp) + mq : on applique ce déplacement à la pose du robot
    x = c * (pose[0] - mp[0]) - s * (pose[1] - mp[1]) + mq[0]
    y = s * (pose[0] - mp[0]) + c * (pose[1] - mp[1]) + mq[1]
    return [x, y, pose[2] + math.degrees(dtheta)]


def calcul_direct(ref, pts, depart):
    """§8.5 : une seule étape, approximation directe de la transformation."""
    return un_calcul_direct(ref, pts, depart)


def icp(ref, pts, depart):
    """Calcul direct répété jusqu'à convergence (Iterative Closest Point) : à chaque tour,
    les correspondances sont meilleures, donc la pose aussi."""
    pose = depart
    for _ in range(IT_MAX):
        nouvelle = un_calcul_direct(ref, pts, pose)
        bouge = math.dist(nouvelle[:2], pose[:2]) + abs(nouvelle[2] - pose[2]) * math.pi / 180
        pose = nouvelle
        if bouge < 1e-4:
            break                                                  # la pose ne bouge plus : convergé
    return pose


def erreur(pose, vraie):
    """Erreur de position (m) et d'angle (°) entre une pose et la vraie pose."""
    return math.dist(pose[:2], vraie[:2]), abs((pose[2] - vraie[2] + 180) % 360 - 180)


if __name__ == "__main__":
    poses = charger_poses("poses.csv")
    vraie = poses.get(SCAN_TEST)                 # None si c'est un nouveau scan, de pose inconnue

    # Référence : grid map construite SANS le scan à localiser
    carte = nouvelle_carte()
    for nom, pose in poses.items():
        if nom != SCAN_TEST:
            mettre_a_jour(carte, lire_scan(nom), pose)
    ref_haute = reference_grille(carte)                         # cellules de 5 cm
    ref_basse = reference_grille(carte, facteur=4, d_max=1.0)   # cellules de 20 cm (§8.2)
    zone = zone_occupee(ref_haute)                              # zone de recherche : là où il y a des murs

    # Scan courant : prétraitement (filtres de lire_scan) + sous-échantillonnage régulier sur le tour
    scan = lire_scan(SCAN_TEST)
    tous = np.array([(r * math.cos(a), r * math.sin(a)) for a, r in scan])
    pts = tous[np.linspace(0, len(tous) - 1, min(NB_POINTS, len(tous))).astype(int)]

    # Estimation initiale pour le gradient (comme une odométrie imprécise)
    if vraie is not None:
        depart = [v + e for v, e in zip(vraie, ERREUR_DEPART)]
    else:
        depart = ESTIMATION_INITIALE

    methodes = [("Multi-résolution (sans estimation initiale)",
                 lambda: multi_resolution(ref_basse, ref_haute, pts, zone))]
    if depart is not None:
        methodes.append(("Gradient (avec estimation initiale)", lambda: descente_gradient(ref_haute, pts, depart)))

    print(f"Scan localisé : {SCAN_TEST}   ({len(pts)} points)")
    if depart is not None:
        print(f"Estimation initiale : x = {depart[0]:.2f}  y = {depart[1]:.2f}  θ = {depart[2]:.1f}°")
    resultats = {}
    for nom, methode in methodes:
        t = time.time()
        pose = methode()
        duree = time.time() - t
        resultats[nom] = pose
        ligne = f"{nom:45s} x = {pose[0]:6.2f}  y = {pose[1]:6.2f}  θ = {pose[2]:7.2f}°   ({duree:.2f} s)"
        if vraie is not None:
            d, a = erreur(pose, vraie)
            ligne += f"   erreur : {d * 100:.1f} cm, {a:.1f}°"
        print(ligne)
    if vraie is not None:
        print(f"{'Pose de poses.csv':45s} x = {vraie[0]:6.2f}  y = {vraie[1]:6.2f}  θ = {vraie[2]:7.2f}°")

    # Affichage : le scan placé à chaque pose trouvée, sur la carte
    d = TAILLE / 2
    fig, axes = plt.subplots(1, len(resultats), figsize=(7 * len(resultats), 7), squeeze=False)
    for ax, (nom, pose) in zip(axes.flat, resultats.items()):
        ax.imshow(carte > INCONNU, cmap="gray_r", origin="lower", extent=[-d, d, -d, d])
        X, Y = transformer(tous, *pose)
        ax.scatter(X, Y, s=1, color="red", label="scan à la pose trouvée")
        if "Gradient" in nom:
            ax.plot(*depart[:2], "bx", label="estimation initiale")
        if vraie is not None:
            ax.plot(*vraie[:2], "g+", markersize=12, label="pose de poses.csv")
        ax.plot(*pose[:2], "r^", label="pose trouvée")
        ax.set_xlim(zone[0][0] - 1, zone[0][1] + 1)
        ax.set_ylim(zone[1][0] - 1, zone[1][1] + 1)
        ax.set_title(nom)
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()