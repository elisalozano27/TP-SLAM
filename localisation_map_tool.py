# localisation_site.py - Compare les 5 méthodes de localisation sur les données du site du prof
# (grid_map.csv + scan_lidar.csv, dont la vraie pose est écrite en 1re ligne du scan).
#
# Test à l'aveugle : la vraie pose n'est lue qu'APRÈS les recherches, pour calculer les erreurs.
#   Test 1 « global » : aucune estimation initiale, on cherche dans toute la carte.
#   Test 2 « local »  : NB_ESSAIS estimations initiales fausses tirées au hasard (comme une odométrie
#                       imprécise) ; pour chaque méthode : erreurs relatives moyennes et temps moyen.
#
# Erreur relative = |valeur trouvée - valeur attendue| / |valeur attendue|, en %, pour x, y et θ.

import math
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import localisation as L

CARTE = "Map_Tool_grid_map.csv"
SCAN = "Map_Tool_scan_lidar.csv"
RES = 0.1                    # taille d'une cellule de la carte du site (m), écrite dans son en-tête

NB_ESSAIS = 20               # test 2 : nb d'estimations initiales tirées au hasard
ECART_XY = 0.40              # test 2 : erreur maximale sur x et sur y (m)
ECART_THETA = 10.0           # test 2 : erreur maximale sur θ (°)


def lire_carte(fichier, ligne0_en_bas):
    """Carte du site : 1 = occupé, 0 = libre. ligne0_en_bas : la 1re ligne du fichier est-elle y = 0 ?"""
    g = np.loadtxt(fichier, delimiter=",", comments="#")
    return g if ligne0_en_bas else g[::-1]


def lire_scan_site(fichier):
    """Points (x, y) du scan dans le repère du robot (colonnes x_rel_m et y_rel_m)."""
    d = np.loadtxt(fichier, delimiter=",", comments="#", skiprows=2)   # on saute le commentaire et l'en-tête
    return d[d[:, 1] > 0][:, 2:4]                                       # r = 0 : pas de retour, ignoré


def lire_vraie_pose(fichier):
    with open(fichier) as f:
        x, y, t = re.findall(r"=\s*(-?[\d.]+)", f.readline())
    return [float(x), float(y), float(t)]


def references(ligne0_en_bas):
    proba = lire_carte(CARTE, ligne0_en_bas)
    return L.reference(proba, RES, (0.0, 0.0)), L.reference(proba, RES, (0.0, 0.0), facteur=2, d_max=1.0)


def erreurs_relatives(pose, vraie):
    """Erreurs relatives (%) sur x, y et θ par rapport à la pose attendue."""
    ecarts = [pose[0] - vraie[0], pose[1] - vraie[1], (pose[2] - vraie[2] + 180) % 360 - 180]
    return [abs(e) / abs(v) * 100 for e, v in zip(ecarts, vraie)]


def chrono(methode):
    t = time.time()
    pose = methode()
    return pose, time.time() - t


if __name__ == "__main__":
    pts = lire_scan_site(SCAN)
    print(len(pts), "points dans le scan")

    # Sens de la carte : on essaie les deux et on garde celui où le scan s'emboîte le mieux (sans la vraie pose)
    essais = []
    for sens in (True, False):
        haute, basse = references(sens)
        p = L.multi_resolution(basse, haute, pts, L.zone_occupee(haute))
        essais.append((L.score(haute, pts, *p), sens))
    sens = max(essais)[1]
    print("Sens de la carte : 1re ligne du fichier =", "bas" if sens else "haut")
    ref_haute, ref_basse = references(sens)
    zone = L.zone_occupee(ref_haute)
    (xmin, xmax), (ymin, ymax) = zone
    centre = [(xmin + xmax) / 2, (ymin + ymax) / 2, 0.0]
    demi = max(xmax - xmin, ymax - ymin) / 2

    # ---------- Test 1 : global, sans estimation initiale ----------
    globales = {
        "Force brute": lambda: L.force_brute(ref_haute, pts, centre, demi, 180, 0.05, 1),
        "Gradient (départ au centre)": lambda: L.descente_gradient(ref_haute, pts, centre),
        "Multi-résolution": lambda: L.multi_resolution(ref_basse, ref_haute, pts, zone),
        "Filtre à particules": lambda: L.filtre_particules(ref_basse, ref_haute, pts, zone),
        "ICP (départ au centre)": lambda: L.icp(ref_haute, pts, centre),
    }
    test1 = {nom: chrono(m) for nom, m in globales.items()}

    # ---------- Test 2 : local, avec des estimations initiales fausses ----------
    rng = np.random.default_rng(1)
    erreurs_depart = rng.uniform(-1, 1, (NB_ESSAIS, 3)) * [ECART_XY, ECART_XY, ECART_THETA]
    locales = {
        "Force brute locale": lambda d: L.force_brute(ref_haute, pts, d, 0.60, 15, 0.05, 1),
        "Gradient": lambda d: L.descente_gradient(ref_haute, pts, d),
        "Calcul direct (1 étape)": lambda d: L.calcul_direct(ref_haute, pts, d),
        "ICP (calcul direct répété)": lambda d: L.icp(ref_haute, pts, d),
    }

    # ---------- Seulement maintenant : la vraie pose ----------
    vraie = lire_vraie_pose(SCAN)
    departs = [[v + e for v, e in zip(vraie, err)] for err in erreurs_depart]
    test2 = {nom: [chrono(lambda: m(d)) for d in departs] for nom, m in locales.items()}

    print(f"\nPose attendue : x = {vraie[0]:.2f} m  y = {vraie[1]:.2f} m  θ = {vraie[2]:.1f}°")
    entete = f"{'Méthode':30s} {'x':>7s} {'y':>7s} {'θ':>7s}   temps"
    print("\nTest 1 : sans estimation initiale — erreurs relatives")
    print(entete)
    for nom, (pose, duree) in test1.items():
        ex, ey, et = erreurs_relatives(pose, vraie)
        print(f"{nom:30s} {ex:6.1f}% {ey:6.1f}% {et:6.1f}%   {duree:5.2f} s")

    print(f"\nTest 2 : {NB_ESSAIS} estimations initiales fausses (jusqu'à ±{ECART_XY * 100:.0f} cm, ±{ECART_THETA:.0f}°)"
          " — erreurs relatives moyennes")
    print(entete)
    ex, ey, et = np.mean([erreurs_relatives(d, vraie) for d in departs], axis=0)
    print(f"{'(estimation initiale)':30s} {ex:6.1f}% {ey:6.1f}% {et:6.1f}%")
    for nom, liste in test2.items():
        ex, ey, et = np.mean([erreurs_relatives(pose, vraie) for pose, _ in liste], axis=0)
        print(f"{nom:30s} {ex:6.1f}% {ey:6.1f}% {et:6.1f}%   {np.mean([t for _, t in liste]) * 1000:5.1f} ms")

    # ---------- Affichage : une pose par méthode ----------
    a_montrer = {nom + " (global)": test1[nom][0] for nom in ("Force brute", "Multi-résolution", "Filtre à particules")}
    a_montrer.update({nom + " (local, essai 1)": test2[nom][0][0]
                      for nom in ("Gradient", "Calcul direct (1 étape)", "ICP (calcul direct répété)")})
    carte = lire_carte(CARTE, sens)
    fig, axes = plt.subplots(2, 3, figsize=(14, 11))
    for ax, (nom, pose) in zip(axes.flat, a_montrer.items()):
        ax.imshow(carte, cmap="gray_r", origin="lower", extent=[0, carte.shape[1] * RES, 0, carte.shape[0] * RES])
        X, Y = L.transformer(pts, *pose)
        ax.scatter(X, Y, s=8, color="red")
        if "local" in nom:
            ax.plot(*departs[0][:2], "bx", label="départ")
        ax.plot(*vraie[:2], "g+", markersize=14, mew=2, label="vraie pose")
        ax.plot(*pose[:2], "r^", label="pose trouvée")
        ax.set_title(nom, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()