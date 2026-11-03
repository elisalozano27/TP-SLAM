# grille.py - Construit et met à jour une carte d'occupation (grid map 2D) à partir de scans du LD06
# Suit les étapes du document « Construction d'une grid map 2D à partir d'un scan Lidar »
#
# Outil 1 : carte_depuis_scan(scan, pose)      -> nouvelle carte
# Outil 2 : mettre_a_jour(carte, scan, pose)   -> carte mise à jour
#
# Chaque cellule contient une probabilité d'occupation entre 0 et 255 :
#   0 = libre certain, 127 = inconnu, 255 = occupé certain

import csv
import math
from bisect import bisect_left, bisect_right
import numpy as np
import matplotlib.pyplot as plt

# ---------- Étape 1 : définir la grille ----------
TAILLE = 28.0                    # côté de la carte (m), centrée sur la position du premier scan
RESOLUTION = 0.05                # côté d'une cellule (m)
N = int(TAILLE / RESOLUTION)     # nombre de cellules par côté (560)
INCONNU = 127                    # état initial de chaque cellule

PLUS_OCC = 50     # ajouté à une cellule d'impact
MOINS_LIBRE = 15  # retiré à une cellule traversée

# Prétraitement : mettre True ou False pour activer ou désactiver chaque filtre
FILTRE_INTENSITE = True          # retirer les retours trop faibles
FILTRE_MEDIANE = True            # corriger les distances par médiane glissante

R_MIN, R_MAX = 0.10, 8.0         # filtrage : distances gardées (m)
INTENSITE_MIN = 100              # filtrage : intensité minimale gardée (0 à 255), à régler
FENETRE = 0.5                    # médiane : demi-largeur de la fenêtre autour de chaque point (°)
MIN_MESURES = 3                  # médiane : nb minimum de mesures dans la fenêtre pour garder le point


def nouvelle_carte():
    """Carte où toutes les cellules sont inconnues."""
    return np.full((N, N), INCONNU, dtype=np.uint8)


# ---------- Étape 2 : points lidar -> coordonnées de la grille ----------
def lire_scan(fichier):
    """Lit tous les tours d'un fichier CSV et applique les filtres activés :
    - FILTRE_INTENSITE : retire les points d'intensité < INTENSITE_MIN ;
    - FILTRE_MEDIANE : chaque point garde son angle exact, mais sa distance est remplacée par
      la médiane des distances mesurées à ±FENETRE° autour de lui (tous tours confondus).
    Renvoie la liste des (angle en rad, distance en m), triée par angle."""
    mesures = []                              # (angle en °, distance en m) de tous les tours
    with open(fichier) as f:
        for ligne in csv.DictReader(f):
            r = float(ligne["distance_mm"]) / 1000
            intensite = float(ligne.get("intensite", 255))   # anciens CSV sans intensité : point gardé
            if not R_MIN <= r <= R_MAX:       # filtrage : pas de retour (0), trop près ou trop loin
                continue
            if FILTRE_INTENSITE and intensite < INTENSITE_MIN:   # retour trop faible, peu fiable
                continue
            mesures.append((float(ligne["angle_deg"]), r))
    mesures.sort()
    if not FILTRE_MEDIANE:                    # sans médiane : on garde les points tels quels
        return [(math.radians(a), r) for a, r in mesures]
    angles = [a for a, r in mesures]
    scan = []
    for a, r in mesures:
        debut = bisect_left(angles, a - FENETRE)     # premier voisin de la fenêtre
        fin = bisect_right(angles, a + FENETRE)      # juste après le dernier voisin
        voisins = [d for _, d in mesures[debut:fin]]
        if len(voisins) >= MIN_MESURES:      # point trop isolé -> pas fiable, ignoré
            scan.append((math.radians(a), float(np.median(voisins))))
    return scan


def charger_poses(fichier="poses.csv"):
    """Lit les poses enregistrées par affichage.py : {nom du scan: [x, y, theta]}."""
    poses = {}
    with open(fichier) as f:
        for ligne in csv.DictReader(f):
            poses[ligne["scan"]] = [float(ligne["x"]), float(ligne["y"]), float(ligne["theta"])]
    return poses


def vers_cellule(x, y):
    """Position globale (m) -> indices (colonne, ligne) dans la grille."""
    return math.floor(x / RESOLUTION) + N // 2, math.floor(y / RESOLUTION) + N // 2


# ---------- Étape 3 : cellules traversées par le faisceau ----------
def bresenham(c0, l0, c1, l1):
    """Liste des cellules sur le segment (c0, l0) -> (c1, l1), extrémités comprises."""
    cellules = []
    dc, dl = abs(c1 - c0), -abs(l1 - l0)
    sc = 1 if c0 < c1 else -1
    sl = 1 if l0 < l1 else -1
    err = dc + dl
    while True:
        cellules.append((c0, l0))
        if c0 == c1 and l0 == l1:
            return cellules
        e2 = 2 * err
        if e2 >= dl:
            err += dl
            c0 += sc
        if e2 <= dc:
            err += dc
            l0 += sl


def dans_carte(c, l):
    return 0 <= c < N and 0 <= l < N


# ---------- Étapes 2 + 3 + 4 pour tout un scan ----------
def mettre_a_jour(carte, scan, pose):
    """Outil 2 : ajoute un scan (pris à la pose [x, y, theta°]) dans une carte existante."""
    x0, y0, theta = pose
    th = math.radians(theta)
    c0, l0 = vers_cellule(x0, y0)               # cellule du robot
    calcul = carte.astype(int)                  # copie en entiers pour pouvoir dépasser 0..255
    for a, r in scan:
        # Étape 2 : repère du robot -> repère global (rotation de theta + translation de x0, y0)
        x = x0 + r * math.cos(th + a)
        y = y0 + r * math.sin(th + a)
        c1, l1 = vers_cellule(x, y)
        # Étape 3 : cellules traversées (sauf la dernière) -> probabilité diminuée
        for c, l in bresenham(c0, l0, c1, l1)[:-1]:
            if dans_carte(c, l):
                calcul[l, c] -= MOINS_LIBRE
        # Étape 4 : cellule d'impact -> probabilité augmentée
        if dans_carte(c1, l1):
            calcul[l1, c1] += PLUS_OCC
    carte[:] = np.clip(calcul, 0, 255)          # on reste entre 0 et 255
    return carte


def carte_depuis_scan(scan, pose):
    """Outil 1 : crée une carte inconnue puis y met un scan."""
    return mettre_a_jour(nouvelle_carte(), scan, pose)


def boucher_trous(carte):
    """Lissage (étape 5) : une cellule inconnue entourée surtout de cellules libres devient libre."""
    libre = (carte < INCONNU).astype(int)
    voisins = sum(np.roll(np.roll(libre, dl, 0), dc, 1)       # nombre de voisines libres (sur 8)
                  for dl in (-1, 0, 1) for dc in (-1, 0, 1)) - libre
    trous = (carte == INCONNU) & (voisins >= 5)
    carte[trous] = INCONNU - MOINS_LIBRE
    return carte


def afficher(carte, titre):
    """Affiche la carte en noir et blanc : noir = occupé, blanc = tout le reste."""
    image = np.zeros(carte.shape)   # libre et non scanné -> blanc
    image[carte > INCONNU] = 255    # plutôt occupé       -> noir
    d = TAILLE / 2
    plt.figure(figsize=(7, 7))
    plt.imshow(image, cmap="gray_r", origin="lower", extent=[-d, d, -d, d], vmin=0, vmax=255)
    plt.title(titre)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")


if __name__ == "__main__":
    POSES = charger_poses("poses.csv")           # poses enregistrées avec affichage.py
    noms = list(POSES)

    # Outil 1 : carte construite à partir du premier scan seul
    carte = carte_depuis_scan(lire_scan(noms[0]), POSES[noms[0]])
    afficher(boucher_trous(carte.copy()), "Carte à partir de " + noms[0])

    # Outil 2 (étape 5) : on met à jour la même carte avec les scans suivants
    for nom in noms[1:]:
        carte = mettre_a_jour(carte, lire_scan(nom), POSES[nom])
        print(nom, "ajouté")
    afficher(boucher_trous(carte.copy()), f"Carte mise à jour avec {len(noms)} scans")

    plt.show()