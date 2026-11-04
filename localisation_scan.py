# localisation_scan.py - Partie 3 sur le vrai robot : localisation entre scans, de proche en proche
# scan_01 = référence, pose (0, 0, 0). Chaque scan est localisé par rapport au scan PRÉCÉDENT,
# puis sa pose dans la carte = pose du précédent + ce déplacement (composition des poses).
# Aucune pose de poses.csv n'entre dans le calcul. Pas d'odométrie : l'estimation initiale des
# méthodes locales est « le robot n'a pas bougé » (0, 0, 0).
# À la fin, on compare à poses.csv (réglé à la main) : c'est un écart, pas une erreur vraie.

import glob
import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import localisation_grid as L
import localisation_scan_simulation as S
from grille import lire_scan, charger_poses

NB_POINTS = 400               # sous-échantillonnage du scan courant (la référence garde tous ses points)
METHODE_SAUVEE = "Multi-résolution (sans estimation)"   # ses poses sont enregistrées dans FICHIER_SORTIE
FICHIER_SORTIE = "poses_scan_matching.csv"


def points(nom):
    """Scan filtré (lire_scan) -> points (x, y) dans le repère du robot."""
    return np.array([(r * math.cos(a), r * math.sin(a)) for a, r in lire_scan(nom)])


ZERO = [0.0, 0.0, 0.0]        # estimation initiale : le robot n'a pas bougé
METHODES = {
    "Multi-résolution (sans estimation)": lambda h, b, p, z: L.multi_resolution(b, h, p, z),
    "Filtre à particules (sans estimation)": lambda h, b, p, z: L.filtre_particules(b, h, p, z),
    "Force brute locale": lambda h, b, p, z: L.force_brute(h, p, ZERO, 1.0, 30, 0.05, 1),
    "Gradient": lambda h, b, p, z: L.descente_gradient(h, p, ZERO),
    "Calcul direct (1 étape)": lambda h, b, p, z: L.calcul_direct(h, p, ZERO),
    "ICP point à point": lambda h, b, p, z: L.icp(h, p, ZERO),
}


if __name__ == "__main__":
    noms = sorted(glob.glob("scan_*.csv"))               # ordre de capture
    nuages = {n: points(n) for n in noms}
    a_la_main = charger_poses("poses.csv") if os.path.exists("poses.csv") else {}
    if noms[0] in a_la_main:                              # au cas où scan_01 n'y serait pas en (0, 0, 0)
        a_la_main = {n: S.pose_relative(a_la_main[noms[0]], p) for n, p in a_la_main.items()}

    trajets = {m: [ZERO] for m in METHODES}              # pose de chaque scan, pour chaque méthode
    temps = {m: 0.0 for m in METHODES}
    for prec, nom in zip(noms, noms[1:]):
        ref_haute = S.reference_scan(nuages[prec])                       # le scan précédent devient
        ref_basse = S.reference_scan(nuages[prec], facteur=4, d_max=1.0)  # une petite grid map (§6)
        zone = L.zone_occupee(ref_haute)
        tous = nuages[nom]
        pts = tous[np.linspace(0, len(tous) - 1, min(NB_POINTS, len(tous))).astype(int)]
        for m, methode in METHODES.items():
            t = time.time()
            rel = methode(ref_haute, ref_basse, pts, zone)
            temps[m] += time.time() - t
            trajets[m].append(S.composer(trajets[m][-1], rel))
        ligne = f"{nom} localisé par rapport à {prec}"
        if prec in a_la_main and nom in a_la_main:
            ligne += f"   (déplacement d'après poses.csv : {math.dist(a_la_main[prec][:2], a_la_main[nom][:2]):.2f} m)"
        print(ligne)

    # Comparaison avec poses.csv : écart (cm / °) pour chaque scan
    if a_la_main:
        print("\nÉcart avec poses.csv (cm / °) — il s'accumule de scan en scan (dérive)")
        print(f"{'Méthode':38s}" + "".join(f"{n[5:7]:>11s}" for n in noms[1:]) + "   temps total")
        for m, trajet in trajets.items():
            ecarts = [L.erreur(p, a_la_main[n]) for p, n in zip(trajet[1:], noms[1:]) if n in a_la_main]
            print(f"{m:38s}" + "".join(f"{d * 100:6.1f}/{a:4.1f}" for d, a in ecarts) + f"   {temps[m]:6.1f} s")

    with open(FICHIER_SORTIE, "w") as f:
        f.write("scan,x,y,theta\n")
        for n, (x, y, th) in zip(noms, trajets[METHODE_SAUVEE]):
            f.write(f"{n},{x:.3f},{y:.3f},{th:.2f}\n")
    print(f"\nPoses trouvées par « {METHODE_SAUVEE} » enregistrées dans {FICHIER_SORTIE}")

    # Affichage : trajectoires, puis tous les scans placés aux poses de METHODE_SAUVEE
        # Affichage 1 : trajectoires de toutes les méthodes
    plt.figure(figsize=(8, 7))
    for m, trajet in trajets.items():
        plt.plot(*np.array(trajet)[:, :2].T, "o-", label=m)
    if a_la_main:
        plt.plot(*np.array([a_la_main[n][:2] for n in noms if n in a_la_main]).T, "k+--", markersize=12,
                 label="poses.csv (à la main)")
    plt.title("Trajectoires obtenues de proche en proche")
    plt.legend(fontsize=8)
    plt.axis("equal")
    plt.grid(True)

    # Affichage 2 : les scans placés avec chaque méthode (une case par méthode)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (m, trajet) in zip(axes.flat, trajets.items()):
        for n, pose in zip(noms, trajet):
            ax.scatter(*L.transformer(nuages[n], *pose), s=1, label=n)
            ax.plot(*pose[:2], "k^")
        ax.set_title(m, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True)
    axes.flat[0].legend(fontsize=7, markerscale=8)
    plt.tight_layout()
    plt.show()