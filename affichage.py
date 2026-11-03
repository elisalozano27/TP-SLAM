# affichage.py - Aligne à la main tous les scans du dossier et enregistre leurs poses dans poses.csv
#
# - Liste à droite : choisir le scan à déplacer
# - Curseurs en bas : x (m), y (m), θ (°) du scan choisi
# - Clavier (réglage fin, cliquer d'abord dans la fenêtre) :
#     flèches = ±1 cm, a / z = ±0,5°, n / b = scan suivant / précédent
# - Bouton « Sauvegarder » : écrit les poses de tous les scans dans poses.csv
# Si poses.csv existe déjà, on repart des poses enregistrées.

import csv
import glob
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from grille import lire_scan, charger_poses

FICHIER_POSES = "poses.csv"
LIMITE_XY = 15.0       # plage des curseurs x et y (m)
PAS_XY = 0.01          # réglage fin au clavier (m)
PAS_THETA = 0.5        # réglage fin au clavier (°)

# Matplotlib utilise déjà certaines touches (flèches...) : on les libère
TOUCHES = ["left", "right", "up", "down", "a", "z", "n", "b"]
for cle in list(plt.rcParams.keys()):
    if cle.startswith("keymap."):
        plt.rcParams[cle] = [k for k in plt.rcParams[cle] if k not in TOUCHES]

# ---------- Chargement des scans et des poses ----------
noms = sorted(glob.glob("scan_*.csv"))
poses = {nom: [0.0, 0.0, 0.0] for nom in noms}
if os.path.exists(FICHIER_POSES):
    poses.update(charger_poses(FICHIER_POSES))     # on reprend le travail déjà fait
points = {}
for nom in noms:
    scan = lire_scan(nom)                          # mêmes filtres que pour les cartes
    points[nom] = np.array([(r * math.cos(a), r * math.sin(a)) for a, r in scan])
    print(nom, ":", len(scan), "points")


def placer(pts, pose):
    """Points du repère du robot -> repère de la carte (rotation de theta puis translation)."""
    x0, y0, theta = pose
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    return pts @ np.array([[c, s], [-s, c]]) + [x0, y0]


# ---------- Fenêtre ----------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_axes([0.05, 0.25, 0.65, 0.70])
couleurs = plt.cm.tab10.colors
nuages, robots = {}, {}
for i, nom in enumerate(noms):
    coul = couleurs[i % 10]
    nuages[nom] = ax.scatter([], [], s=1, color=coul)
    robots[nom], = ax.plot([], [], "-o", color=coul, markevery=[0])   # position + direction du robot

liste = RadioButtons(fig.add_axes([0.74, 0.35, 0.24, 0.60]), noms)
for i, texte in enumerate(liste.labels):
    texte.set_color(couleurs[i % 10])
curseurs = [
    Slider(fig.add_axes([0.10, 0.15, 0.55, 0.03]), "x (m)", -LIMITE_XY, LIMITE_XY, valinit=0, valstep=PAS_XY),
    Slider(fig.add_axes([0.10, 0.10, 0.55, 0.03]), "y (m)", -LIMITE_XY, LIMITE_XY, valinit=0, valstep=PAS_XY),
    Slider(fig.add_axes([0.10, 0.05, 0.55, 0.03]), "θ (°)", -180, 180, valinit=0, valstep=PAS_THETA),
]
bouton = Button(fig.add_axes([0.76, 0.08, 0.20, 0.07]), "Sauvegarder")

selection = noms[0]
chargement = False     # True pendant qu'on place les curseurs sur un autre scan


def redessiner():
    """Met à jour tous les scans ; le scan choisi est bien visible, les autres plus pâles."""
    for nom in noms:
        nuages[nom].set_offsets(placer(points[nom], poses[nom]))
        nuages[nom].set_alpha(1.0 if nom == selection else 0.3)
        x0, y0, theta = poses[nom]
        dx, dy = 0.3 * math.cos(math.radians(theta)), 0.3 * math.sin(math.radians(theta))
        robots[nom].set_data([x0, x0 + dx], [y0, y0 + dy])
    ax.set_title("Scan choisi : " + selection)
    fig.canvas.draw_idle()


def choisir(nom):
    """Nouveau scan choisi : les curseurs prennent sa pose."""
    global selection, chargement
    selection = nom
    chargement = True                    # sinon chaque curseur écraserait la pose pendant le changement
    for curseur, valeur in zip(curseurs, poses[nom]):
        curseur.set_val(valeur)
    chargement = False
    redessiner()


def curseur_bouge(valeur):
    """Un curseur a bougé : on met à jour la pose du scan choisi."""
    if not chargement:
        poses[selection] = [c.val for c in curseurs]
        redessiner()


def touche(event):
    """Réglage fin au clavier."""
    pas = {"left": (0, -PAS_XY), "right": (0, PAS_XY), "up": (1, PAS_XY), "down": (1, -PAS_XY),
           "a": (2, PAS_THETA), "z": (2, -PAS_THETA)}
    if event.key in pas:
        i, d = pas[event.key]
        curseurs[i].set_val(curseurs[i].val + d)
    elif event.key in ("n", "b"):
        i = noms.index(selection) + (1 if event.key == "n" else -1)
        liste.set_active(i % len(noms))          # déclenche choisir()


def sauvegarder(event):
    """Écrit les poses de tous les scans dans poses.csv."""
    with open(FICHIER_POSES, "w", newline="") as f:
        f.write("scan,x,y,theta\n")
        for nom in noms:
            x0, y0, theta = poses[nom]
            f.write(f"{nom},{x0:.3f},{y0:.3f},{theta:.2f}\n")
    print("Poses enregistrées dans", FICHIER_POSES)
    ax.set_title(f"Poses enregistrées dans {FICHIER_POSES}")
    fig.canvas.draw_idle()


liste.on_clicked(choisir)
for curseur in curseurs:
    curseur.on_changed(curseur_bouge)
bouton.on_clicked(sauvegarder)
fig.canvas.mpl_connect("key_press_event", touche)

# Premier affichage, puis on ajuste le cadre à tous les points (on peut zoomer avec la loupe)
choisir(noms[0])
tous = np.vstack([placer(points[nom], poses[nom]) for nom in noms])
ax.update_datalim(tous)
ax.autoscale_view()
ax.set_aspect("equal")
ax.margins(0.1)
ax.grid(True)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.show()