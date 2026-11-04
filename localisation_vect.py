# localisation_vectorielle_map_tool.py - Compare les méthodes de localisation sur la carte VECTORIELLE
# du Map Tool du prof (liste de segments) + scan dont la vraie pose est écrite en 1re ligne.
# Même démarche que localisation_map_tool.py (grid map) : seul le score change (§5 du document).
#
# Test à l'aveugle : la vraie pose n'est lue qu'APRÈS les recherches, pour calculer les erreurs.
#   Test 1 « global » : aucune estimation initiale, on cherche dans toute la carte.
#   Test 2 « local »  : NB_ESSAIS estimations initiales fausses tirées au hasard.
#
# Erreur relative = |valeur trouvée - valeur attendue| / |valeur attendue|, en %, pour x, y et θ.

import math
import numpy as np
import matplotlib.pyplot as plt
import localisation_grid as L
from localisation_map_tool import lire_scan_site, lire_vraie_pose, erreurs_relatives, chrono

CARTE = "Map_Tool_vect.txt"          # segments x1,y1,x2,y2 en mètres
SCAN = "Map_Tool_scan_lidar.csv"     # scan pris dans cette carte

NB_ESSAIS = 20               # test 2 : nb d'estimations initiales tirées au hasard
ECART_XY = 0.40              # test 2 : erreur maximale sur x et sur y (m)
ECART_THETA = 10.0           # test 2 : erreur maximale sur θ (°)
NB_ESSAIS_NIVEAU = 10        # test 3 : nb d'estimations initiales par niveau d'erreur
SEUIL_XY = 0.05              # réussite : erreur de position < SEUIL_XY (m)
SEUIL_THETA = 2.0            # ... et erreur angulaire < SEUIL_THETA (°)


def lire_carte_vect(fichier):
    """Liste des segments ((x1, y1), (x2, y2)) d'un fichier texte x1,y1,x2,y2 (lignes # ignorées)."""
    s = np.loadtxt(fichier, delimiter=",", comments="#", ndmin=2)
    return [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in s[:, :4]]


if __name__ == "__main__":
    segments = lire_carte_vect(CARTE)
    pts = lire_scan_site(SCAN)
    print(f"Carte : {len(segments)} segments   scan : {len(pts)} points")
    ref_haute = L.reference_vectorielle(segments, L.D_MAX)
    ref_basse = L.reference_vectorielle(segments, 2 * L.D_MAX)   # « basse résolution » : score plus large (§8.2)
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
    entete = f"{'Méthode':30s} {'x':>7s} {'y':>7s} {'θ':>7s}   {'erreur (cm, °)':>16s}   temps"
    print("\nTest 1 : sans estimation initiale — erreurs relatives")
    print(entete)
    for nom, (pose, duree) in test1.items():
        ex, ey, et = erreurs_relatives(pose, vraie)
        d, a = L.erreur(pose, vraie)
        print(f"{nom:30s} {ex:6.1f}% {ey:6.1f}% {et:6.1f}%   {d * 100:8.1f} cm {a:5.2f}°   {duree:5.2f} s")

    print(f"\nTest 2 : {NB_ESSAIS} estimations initiales fausses (jusqu'à ±{ECART_XY * 100:.0f} cm, ±{ECART_THETA:.0f}°)"
          f" — erreurs moyennes et taux de réussite (< {SEUIL_XY * 100:.0f} cm et < {SEUIL_THETA:.0f}°)")
    print(entete)
    ex, ey, et = np.mean([erreurs_relatives(d, vraie) for d in departs], axis=0)
    dm, am = np.mean([L.erreur(d, vraie) for d in departs], axis=0)
    print(f"{'(estimation initiale)':30s} {ex:6.1f}% {ey:6.1f}% {et:6.1f}%   {dm * 100:8.1f} cm {am:5.2f}°")
    for nom, liste in test2.items():
        ex, ey, et = np.mean([erreurs_relatives(pose, vraie) for pose, _ in liste], axis=0)
        err = [L.erreur(pose, vraie) for pose, _ in liste]
        dm, am = np.mean(err, axis=0)
        taux = 100 * np.mean([d < SEUIL_XY and a < SEUIL_THETA for d, a in err])
        print(f"{nom:30s} {ex:6.1f}% {ey:6.1f}% {et:6.1f}%   {dm * 100:8.1f} cm {am:5.2f}°"
              f"   {np.mean([t for _, t in liste]) * 1000:5.1f} ms   réussite {taux:3.0f}%")

    # ---------- Test 3 : sensibilité à l'estimation initiale (erreur de départ de plus en plus grande) ----------
    import localisation_scan_simulation as S            # ici : sinon les imports tournent en rond
    res = {}
    for k, (exy, eth) in enumerate(S.ERREURS_DEPART):
        for _ in range(NB_ESSAIS_NIVEAU):
            phi = rng.uniform(0, 2 * math.pi)            # direction de l'erreur tirée au hasard
            d = [vraie[0] + exy * math.cos(phi), vraie[1] + exy * math.sin(phi), vraie[2] + eth * rng.choice([-1, 1])]
            for nom, m in locales.items():
                pose, t = chrono(lambda: m(d))
                res.setdefault(nom, [[] for _ in S.ERREURS_DEPART])[k].append((*L.erreur(pose, vraie), t))
    S.SEUIL_XY, S.SEUIL_THETA = SEUIL_XY, SEUIL_THETA   # afficher lit les seuils dans localisation_scan_simulation
    S.afficher(res, f"Test 3 : {NB_ESSAIS_NIVEAU} départs faux par niveau")

    # ---------- Affichage : une pose par méthode ----------
    a_montrer = {nom + " (global)": test1[nom][0] for nom in ("Force brute", "Multi-résolution", "Filtre à particules")}
    a_montrer.update({nom + " (local, essai 1)": test2[nom][0][0]
                      for nom in ("Gradient", "Calcul direct (1 étape)", "ICP (calcul direct répété)")})
    fig, axes = plt.subplots(2, 3, figsize=(14, 11))
    for ax, (nom, pose) in zip(axes.flat, a_montrer.items()):
        for (x1, y1), (x2, y2) in segments:
            ax.plot([x1, x2], [y1, y2], "k-", linewidth=2)
        X, Y = L.transformer(pts, *pose)
        ax.scatter(X, Y, s=8, color="red")
        if "local" in nom:
            ax.plot(*departs[0][:2], "bx", label="départ")
        ax.plot(*vraie[:2], "g+", markersize=14, mew=2, label="vraie pose")
        ax.plot(*pose[:2], "r^", label="pose trouvée")
        ax.set_aspect("equal")
        ax.set_title(nom, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()