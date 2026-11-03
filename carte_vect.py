# carte_vectorielle.py - Construit et met à jour une carte vectorielle 2D (segments) à partir de scans du LD06
# Suit les étapes du document « Construction d'une carte vectorielle 2D à partir d'un scan lidar »
# Réutilise lire_scan (grille.py) et les fonctions de segments.py
#
# La carte est une liste de features. Une feature = un segment + ses propriétés :
#   {"seg": ((x1, y1), (x2, y2)), "confiance": nb de scans qui l'ont vue, "vu": n° du dernier scan qui l'a vue}

import math
import matplotlib.pyplot as plt
from grille import lire_scan, charger_poses
from segment import (nuage_vers_segments, sont_proches, fusionner, intersection, decouper,
                      longueur, angle_entre, parametres, distance_point_segment, LONGUEUR_MIN)

# Étape 5 : absences
MARGE = 0.20           # on raccourcit chaque rayon de MARGE avant son impact, pour ne pas "traverser" le mur touché (m)
TRAVERSEES_MIN = 3     # nb de rayons qui doivent traverser une feature pour la juger absente
# Étape 6 : nettoyage
ANGLE_COIN = 30.0      # angle minimal entre deux murs pour former un coin (°)
COIN_MAX = 0.30        # distance maximale entre l'extrémité d'un mur et le coin, pour la déplacer dessus (m)
AGE_MAX = 1000         # une feature vue une seule fois et pas revue depuis AGE_MAX scans est supprimée
                       # (1000 = jamais : la salle ne change pas pendant les scans)


# ---------- Étape 1 : la carte ----------
def nouvelle_carte():
    """Par défaut, la carte est vide."""
    return []


def enregistrer(carte, fichier):
    """Enregistre la carte dans un fichier texte : une feature par ligne."""
    with open(fichier, "w") as f:
        f.write("# Carte vectorielle (segments)\n# x1,y1,x2,y2,confiance,dernier_scan\n")
        for feat in carte:
            (x1, y1), (x2, y2) = feat["seg"]
            f.write(f"{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{feat['confiance']},{feat['vu']}\n")


# ---------- Étape 2 : nuage de points global ----------
def vers_global(scan, pose):
    """(angle, distance) dans le repère du robot -> points (x, y) dans le repère de la carte."""
    x0, y0, theta = pose
    th = math.radians(theta)
    return [(x0 + r * math.cos(th + a), y0 + r * math.sin(th + a)) for a, r in scan]


# ---------- Étape 4 : ajout ou fusion ----------
def ajouter(carte, seg, k):
    """Ajoute le segment seg vu au scan n° k : fusion avec une feature proche, sinon nouvelle feature."""
    for feat in carte:
        if sont_proches(feat["seg"], seg):
            feat["seg"] = fusionner(feat["seg"], seg)
            if feat["vu"] != k:               # la confiance compte les scans, pas les morceaux
                feat["confiance"] += 1
                feat["vu"] = k
            return
    carte.append({"seg": seg, "confiance": 1, "vu": k})


# ---------- Étape 5 : absences ----------
def gerer_absences(carte, points, pose, k):
    """Une feature traversée par au moins TRAVERSEES_MIN rayons est probablement absente (porte ouverte,
    meuble déplacé) : sa confiance baisse et elle est réduite à son plus long côté.
    Les features vues par ce scan n° k ne sont pas testées : elles sont forcément là."""
    x0, y0, _ = pose
    traversees = [[] for _ in carte]          # rayons qui traversent chaque feature
    for p in points:
        d = math.dist((x0, y0), p)
        if d <= MARGE:
            continue
        c = (d - MARGE) / d                   # rayon robot -> point, raccourci de MARGE
        rayon = ((x0, y0), (x0 + c * (p[0] - x0), y0 + c * (p[1] - y0)))
        for i, feat in enumerate(carte):
            if feat["vu"] == k:               # vue par ce scan : elle est forcément là
                continue
            if intersection(rayon, feat["seg"]) is not None:
                traversees[i].append(rayon)
    for feat, rayons in zip(carte, traversees):
        if len(rayons) >= TRAVERSEES_MIN:
            feat["confiance"] -= 1
            for rayon in rayons:
                feat["seg"] = decouper(feat["seg"], rayon)


# ---------- Étape 6 : nettoyage ----------
def fermer_coins(carte):
    """Deux murs presque perpendiculaires dont les droites se croisent tout près de leurs extrémités
    forment un coin : on déplace l'extrémité la plus proche de chaque mur sur le point de croisement.
    Ça prolonge un mur qui s'arrête trop tôt et ça coupe celui qui dépasse.
    Pour un T (mur qui arrive au milieu d'un autre), seul le mur qui arrive bouge."""
    for f in carte:
        for g in carte:
            if f is g or angle_entre(f["seg"], g["seg"]) < ANGLE_COIN:
                continue
            p = parametres(f["seg"], g["seg"])     # croisement des deux droites (pas des segments)
            if p is None:
                continue
            (ax, ay), (bx, by) = A, B = f["seg"]
            coin = (ax + p[0] * (bx - ax), ay + p[0] * (by - ay))
            if distance_point_segment(coin, g["seg"]) > COIN_MAX:
                continue                           # le coin est loin de l'autre mur
            if math.dist(coin, A) <= math.dist(coin, B):
                if math.dist(coin, A) <= COIN_MAX:
                    f["seg"] = (coin, B)
            elif math.dist(coin, B) <= COIN_MAX:
                f["seg"] = (A, coin)



def nettoyer(carte, k):
    # 1. Fusion des features redondantes (devenues proches après les mises à jour)
    i = 0
    while i < len(carte):
        for j in range(i + 1, len(carte)):
            if sont_proches(carte[i]["seg"], carte[j]["seg"]):
                autre = carte.pop(j)
                carte[i]["seg"] = fusionner(carte[i]["seg"], autre["seg"])
                carte[i]["confiance"] = max(carte[i]["confiance"], autre["confiance"])
                carte[i]["vu"] = max(carte[i]["vu"], autre["vu"])
                break                             # carte[i] a changé : on recommence ses tests
        else:
            i += 1
    # 2. Collisions géométriques : on ferme les coins
    fermer_coins(carte)
    # 3. Suppression des features trop petites, sans confiance, ou vues une seule fois il y a longtemps
    carte[:] = [f for f in carte
                if longueur(f["seg"]) >= LONGUEUR_MIN and f["confiance"] > 0
                and not (f["confiance"] == 1 and k - f["vu"] > AGE_MAX)]


# ---------- Mise à jour complète avec un scan ----------
def mettre_a_jour(carte, scan, pose, k):
    """Ajoute le scan n° k (pris à la pose [x, y, theta°]) dans la carte."""
    points = vers_global(scan, pose)              # étape 2
    for seg in nuage_vers_segments(points, pose[:2]):   # étape 3 : segmentation + régression linéaire
        ajouter(carte, seg, k)                    # étape 4
    gerer_absences(carte, points, pose, k)        # étape 5
    nettoyer(carte, k)                            # étape 6
    return carte


def afficher(carte, poses, titre):
    """Trait plus épais = feature vue par plus de scans."""
    plt.figure(figsize=(7, 7))
    for feat in carte:
        (x1, y1), (x2, y2) = feat["seg"]
        plt.plot([x1, x2], [y1, y2], "k-", linewidth=1 + feat["confiance"])
    for x0, y0, _ in poses:
        plt.plot(x0, y0, "r^")
    plt.axis("equal")
    plt.grid(True)
    plt.title(titre)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")


if __name__ == "__main__":
    POSES = charger_poses("poses.csv")           # poses enregistrées avec affichage.py

    carte = nouvelle_carte()
    for k, nom in enumerate(POSES, start=1):
        mettre_a_jour(carte, lire_scan(nom), POSES[nom], k)
        print(f"Après {nom} : {len(carte)} features")

    enregistrer(carte, "carte_vectorielle.txt")
    print("Carte enregistrée dans carte_vectorielle.txt")
    afficher(carte, POSES.values(), f"Carte vectorielle : {len(carte)} segments")
    plt.show()