# localisation_scans_simulation.py - Partie 3 avec le Map Tool : localiser un scan par rapport à un scan de référence
# Référence : SCAN_REF, dont la pose est connue (écrite en 1re ligne). Scan courant : SCAN, pris ailleurs.
# On cherche où était le robot au scan courant PAR RAPPORT au robot au scan de référence (pose relative),
# puis sa pose dans la salle = pose de référence + pose relative. La carte n'est pas utilisée : seulement les 2 scans.
# Le scan de référence devient une petite grid map (§6) ; pour l'ICP, correspondances point à point.
#
# Test à l'aveugle : la vraie pose du scan courant n'est lue qu'APRÈS les recherches, pour les erreurs.
#   Test 1 : sans estimation initiale (gradient et ICP partent de « le robot n'a pas bougé »)
#   Test 2 : NB_ESSAIS estimations initiales fausses par niveau d'erreur -> taux de réussite

import math
import numpy as np
import matplotlib.pyplot as plt
import localisation_grid as L
from localisation_map_tool import lire_scan_site, lire_vraie_pose, chrono

SCAN_REF = "Map_Tool_scan_lidar.csv"    # scan de référence (pose connue)
SCAN = "Map_Tool_scan_lidar_21.csv"      # scan à localiser par rapport à la référence

RES = 0.05                # cellules de la petite grid map du scan de référence (m)
SAUT_MAX = 0.30           # deux points voisins à moins de SAUT_MAX sont reliés (même mur) (m)
PAS_DENSE = 0.02          # espacement des points ajoutés entre deux points reliés (m)
NB_ESSAIS = 10            # test 2 : nb d'estimations initiales fausses par niveau d'erreur
SEUIL_XY, SEUIL_THETA = 0.05, 2.0                      # critère de réussite
ERREURS_DEPART = [(0.05, 2), (0.10, 4), (0.20, 8), (0.30, 12), (0.40, 16), (0.50, 20)]   # (m, °)
ZERO = [0.0, 0.0, 0.0]    # « le robot n'a pas bougé »


# ---------- Poses relatives ----------
def pose_relative(ref, pose):
    """Pose exprimée dans le repère du robot à la pose ref."""
    dx, dy = pose[0] - ref[0], pose[1] - ref[1]
    c, s = math.cos(math.radians(ref[2])), math.sin(math.radians(ref[2]))
    return [c * dx + s * dy, -s * dx + c * dy, (pose[2] - ref[2] + 180) % 360 - 180]


def composer(ref, rel):
    """Inverse de pose_relative : pose dans la salle à partir de ref et de la pose relative rel."""
    c, s = math.cos(math.radians(ref[2])), math.sin(math.radians(ref[2]))
    return [ref[0] + c * rel[0] - s * rel[1], ref[1] + s * rel[0] + c * rel[1], ref[2] + rel[2]]


# ---------- Le scan de référence devient une petite grid map (§6) ----------
def relier(pts):
    """Le Map Tool ne donne que 100 points (un tous les 3,6°) : on relie deux points voisins proches
    par des points tous les PAS_DENSE, pour que les murs soient continus dans la grid map."""
    pts = pts[np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))]          # ordre du balayage
    dense = [pts]
    for p, q in zip(pts, np.roll(pts, -1, axis=0)):
        n = int(np.linalg.norm(q - p) / PAS_DENSE)
        if np.linalg.norm(q - p) < SAUT_MAX and n >= 2:
            dense.append(p + np.arange(1, n)[:, None] / n * (q - p))
    return np.vstack(dense)


def reference_scan(pts_ref, facteur=1, d_max=L.D_MAX):
    """Cellule occupée (1) si un point du scan de référence y tombe, 0 sinon. Marge de 1 m autour."""
    pts_ref = relier(pts_ref)
    origine = pts_ref.min(axis=0) - 1.0
    nc, nl = np.ceil((pts_ref.max(axis=0) + 1.0 - origine) / RES).astype(int)
    proba = np.zeros((nl, nc))
    c, l = ((pts_ref - origine) / RES).astype(int).T
    proba[l, c] = 1
    ref = L.reference(proba, RES, origine, facteur, d_max)
    ref["cellules"] = pts_ref        # ICP point à point : on associe aux points eux-mêmes, pas aux cellules
    return ref


# ---------- Résultats du test 2 ----------
def afficher(res, titre):
    """Tableaux (taux de réussite, erreurs médianes, temps) et courbe du taux de réussite."""
    noms_err = [f"{e * 100:.0f}cm/{a:.0f}°" for e, a in ERREURS_DEPART]
    print(f"\n{titre} — taux de réussite (< {SEUIL_XY * 100:.0f} cm et < {SEUIL_THETA:.0f}°) selon l'erreur de départ")
    print(f"{'Méthode':30s}" + "".join(f"{n:>11s}" for n in noms_err) + "   temps moyen")
    taux = {}
    for nom, listes in res.items():
        taux[nom] = [100 * np.mean([d < SEUIL_XY and a < SEUIL_THETA for d, a, _ in l]) for l in listes]
        temps = np.mean([t for l in listes for _, _, t in l]) * 1000
        print(f"{nom:30s}" + "".join(f"{v:10.0f}%" for v in taux[nom]) + f"   {temps:7.1f} ms")
    print(f"\n{titre} — erreur médiane (cm / °)")
    for nom, listes in res.items():
        print(f"{nom:30s}" + "".join(f"{np.median([d for d, _, _ in l]) * 100:6.1f}/{np.median([a for _, a, _ in l]):4.1f}"
                                     for l in listes))
    plt.figure(figsize=(8, 5))
    for nom, v in taux.items():
        plt.plot(noms_err, v, "o-", label=nom)
    plt.ylim(-5, 105)
    plt.xlabel("erreur de l'estimation initiale")
    plt.ylabel("taux de réussite (%)")
    plt.title(titre)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()


if __name__ == "__main__":
    pose_ref = lire_vraie_pose(SCAN_REF)             # la pose de la référence est connue
    pts_ref, pts = lire_scan_site(SCAN_REF), lire_scan_site(SCAN)
    print(f"Référence : {len(pts_ref)} points, pose x = {pose_ref[0]:.2f}  y = {pose_ref[1]:.2f}  θ = {pose_ref[2]:.1f}°")
    print(f"Scan à localiser : {len(pts)} points")
    ref_haute = reference_scan(pts_ref)
    ref_basse = reference_scan(pts_ref, facteur=4, d_max=1.0)
    zone = L.zone_occupee(ref_haute)

    # ---------- Test 1 : sans estimation initiale ----------
    globales = {
        "Multi-résolution": lambda: L.multi_resolution(ref_basse, ref_haute, pts, zone),
        "Filtre à particules": lambda: L.filtre_particules(ref_basse, ref_haute, pts, zone),
        "Gradient (départ 0, 0, 0)": lambda: L.descente_gradient(ref_haute, pts, ZERO),
        "ICP (départ 0, 0, 0)": lambda: L.icp(ref_haute, pts, ZERO),
    }
    test1 = {nom: chrono(m) for nom, m in globales.items()}

    # ---------- Test 2 : erreurs de départ tirées AVANT de connaître la vraie pose ----------
    rng = np.random.default_rng(1)
    tirages = [[(rng.uniform(0, 2 * math.pi), rng.choice([-1, 1])) for _ in range(NB_ESSAIS)] for _ in ERREURS_DEPART]

    # ---------- Seulement maintenant : la vraie pose ----------
    vraie_salle = lire_vraie_pose(SCAN)
    vraie = pose_relative(pose_ref, vraie_salle)
    print(f"\nPose relative attendue : x = {vraie[0]:.3f} m  y = {vraie[1]:.3f} m  θ = {vraie[2]:.2f}°"
          f"   (dans la salle : {vraie_salle[0]:.2f}, {vraie_salle[1]:.2f}, {vraie_salle[2]:.1f}°)")
    print("\nTest 1 : sans estimation initiale")
    print(f"{'Méthode':28s} {'pose relative trouvée':>26s}   {'pose dans la salle':>22s}   erreur      temps")
    for nom, (rel, duree) in test1.items():
        x, y, t = composer(pose_ref, rel)
        d, a = L.erreur(rel, vraie)
        print(f"{nom:28s} {rel[0]:7.3f} {rel[1]:7.3f} {rel[2]:8.2f}°   {x:6.2f} {y:6.2f} {t:7.1f}°"
              f"   {d * 100:5.1f} cm {a:4.1f}°   {duree:5.2f} s")

    res = {}
    for k, (exy, eth) in enumerate(ERREURS_DEPART):
        for phi, signe in tirages[k]:
            d = [vraie[0] + exy * math.cos(phi), vraie[1] + exy * math.sin(phi), vraie[2] + eth * signe]
            locales = {"Force brute locale": lambda: L.force_brute(ref_haute, pts, d, exy + 0.10, eth + 2, 0.05, 1),
                       "Gradient": lambda: L.descente_gradient(ref_haute, pts, d),
                       "Calcul direct (1 étape)": lambda: L.calcul_direct(ref_haute, pts, d),
                       "ICP point à point": lambda: L.icp(ref_haute, pts, d)}
            for nom, m in locales.items():
                pose, t = chrono(m)
                res.setdefault(nom, [[] for _ in ERREURS_DEPART])[k].append((*L.erreur(pose, vraie), t))
    afficher(res, f"Test 2 : {NB_ESSAIS} départs faux par niveau")

    # ---------- Affichage du test 1 : les deux scans placés dans la salle ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, (nom, (rel, _)) in zip(axes.flat, test1.items()):
        pose = composer(pose_ref, rel)
        ax.scatter(*L.transformer(pts_ref, *pose_ref), s=10, color="blue", label="référence")
        ax.scatter(*L.transformer(pts, *pose), s=10, color="red", label="scan à la pose trouvée")
        ax.plot(*pose_ref[:2], "b^", label="pose de référence")
        ax.plot(*vraie_salle[:2], "g+", markersize=14, mew=2, label="vraie pose")
        ax.plot(*pose[:2], "r^", label="pose trouvée")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title(nom, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.show()