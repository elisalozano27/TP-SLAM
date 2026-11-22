# rejouer_slam.py - Rejoue un dossier de scans dans la boucle SLAM de robot_slam.py, sans robot ni WiFi
# Même localisation (multi-résolution, recherche à moins de RAYON_RECHERCHE de la pose précédente),
# même mise à jour de la grid map (grille.py). Si le dossier contient poses_vraies_simulation.csv,
# l'erreur de chaque pose est calculée et tracée.
#
# Lancer : python rejouer_slam.py captures_102

import csv
import glob
import math
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import grille as G
import localisation_grid as L

RAYON_RECHERCHE = 2.0   # mêmes réglages que robot_slam.py
NB_POINTS = 400
SCORE_MIN = 0.40


def points(scan):
    tous = np.array([(r * math.cos(a), r * math.sin(a)) for a, r in scan])
    return tous[np.linspace(0, len(tous) - 1, min(NB_POINTS, len(tous))).astype(int)]


def localiser(carte, scan, autour):
    """Comme robot_slam.localiser : multi-résolution autour de la pose précédente. Renvoie (pose, score)."""
    ref_haute = L.reference_grille(carte)
    ref_basse = L.reference_grille(carte, facteur=4, d_max=1.0)
    x, y = autour[0], autour[1]
    zone = ((x - RAYON_RECHERCHE, x + RAYON_RECHERCHE), (y - RAYON_RECHERCHE, y + RAYON_RECHERCHE))
    pts = points(scan)
    x, y, th = L.multi_resolution(ref_basse, ref_haute, pts, zone)
    return [float(x), float(y), float((th + 180) % 360 - 180)], float(L.score(ref_haute, pts, x, y, th))


if __name__ == "__main__":
    dossier = sys.argv[1] if len(sys.argv) > 1 else "captures_102"
    noms = sorted(glob.glob(os.path.join(dossier, "scan_*.csv")))
    carte = G.nouvelle_carte()
    poses, scores = [], []
    for k, fichier in enumerate(noms):
        scan = G.lire_scan(fichier)
        t = time.time()
        if k == 0:
            pose, score = [0.0, 0.0, 0.0], None          # 1er scan = référence
        else:
            pose, score = localiser(carte, scan, poses[-1])
        G.mettre_a_jour(carte, scan, pose)
        poses.append(pose)
        scores.append(score)
        douteux = "  (score faible)" if score is not None and score < SCORE_MIN else ""
        print(f"{os.path.basename(fichier)}  x = {pose[0]:6.2f}  y = {pose[1]:6.2f}  θ = {pose[2]:7.1f}°"
              f"  score {'-' if score is None else f'{score:.2f}'}  ({time.time() - t:.2f} s){douteux}")

    with open(os.path.join(dossier, "poses_slam.csv"), "w") as f:
        f.write("scan,x,y,theta\n")
        for n, (x, y, th) in zip(noms, poses):
            f.write(f"{os.path.basename(n)},{x:.3f},{y:.3f},{th:.2f}\n")
    print("Poses enregistrées dans", os.path.join(dossier, "poses_slam.csv"))

    P = np.array(poses)
    d = G.TAILLE / 2
    plt.figure(figsize=(10, 8))
    plt.imshow(255 - G.boucher_trous(carte.copy()), cmap="gray", origin="lower", extent=[-d, d, -d, d], vmin=0, vmax=255)
    plt.plot(P[:, 0], P[:, 1], "o-", color="tab:blue", ms=3, lw=1, label="trajectoire trouvée")

    vraies_f = os.path.join(dossier, "poses_vraies_simulation.csv")
    if os.path.exists(vraies_f):
        with open(vraies_f) as f:
            V = {r["scan"]: [float(r["x"]), float(r["y"]), float(r["theta"])] for r in csv.DictReader(f)}
        vraies = np.array([V[os.path.basename(n)] for n in noms])
        plt.plot(vraies[:, 0], vraies[:, 1], "--", color="tab:green", label="vraie trajectoire")
        e = [math.dist(p[:2], v[:2]) * 100 for p, v in zip(P, vraies)]
        a = [abs((p[2] - v[2] + 180) % 360 - 180) for p, v in zip(P, vraies)]
        print(f"Erreur de position : moyenne {np.mean(e[1:]):.1f} cm, max {max(e):.1f} cm ;"
              f" erreur d'angle : moyenne {np.mean(a[1:]):.2f}°, max {max(a):.2f}°")
    lo, hi = P[:, :2].min(axis=0) - 2, P[:, :2].max(axis=0) + 2
    plt.xlim(lo[0], hi[0])
    plt.ylim(lo[1], hi[1])
    plt.legend()
    plt.title(f"Carte construite en rejouant {len(noms)} scans")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()
