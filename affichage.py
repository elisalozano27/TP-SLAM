# visu.py - Affiche plusieurs scans du lidar LD06 en coordonnées cartésiennes
#
# Commandes clavier (cliquer d'abord dans la fenêtre) :
#   1, 2, 3...  : choisir le scan à déplacer
#   flèches     : déplacer le scan choisi
#   a / z       : tourner le scan choisi (a = sens antihoraire, z = sens horaire)
# Les poses finales sont affichées dans la console à la fermeture de la fenêtre.

import csv
import math
import matplotlib.pyplot as plt

# Scans à afficher et pose d'origine de chacun : [x (m), y (m), theta (°)]
SCANS = {
    "scan_01.csv": [0.0, 0.0, 0.0],
    "scan_02.csv": [0.0, 0.0, 0.0],
}

PAS_XY = 0.05    # déplacement à chaque appui (m)
PAS_THETA = 2.0  # rotation à chaque appui (°)

# Matplotlib utilise déjà certaines touches (flèches, a...) : on les libère
TOUCHES = ["left", "right", "up", "down", "a", "z"]
for cle in list(plt.rcParams.keys()):
    if cle.startswith("keymap."):
        plt.rcParams[cle] = [k for k in plt.rcParams[cle] if k not in TOUCHES]


def lire_scan(fichier):
    """Lit un fichier CSV et renvoie les points (x, y) dans le repère du robot, en mètres."""
    xs, ys = [], []
    with open(fichier) as f:
        for ligne in csv.DictReader(f):
            r = float(ligne["distance_mm"]) / 1000   # mm -> m
            if r == 0:
                continue                             # 0 = rien détecté, on ignore
            a = math.radians(float(ligne["angle_deg"]))
            xs.append(r * math.cos(a))               # polaire -> cartésien
            ys.append(r * math.sin(a))
    return xs, ys


def placer(xs, ys, pose):
    """Place les points dans la carte selon la pose du robot (rotation puis translation)."""
    x0, y0, theta = pose
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    return [(x0 + c * x - s * y, y0 + s * x + c * y) for x, y in zip(xs, ys)]


# ---------- Chargement des scans ----------
noms = list(SCANS.keys())
points = {nom: lire_scan(nom) for nom in noms}

# Vérification : nombre de points lus dans chaque fichier
for nom in noms:
    print(nom, ":", len(points[nom][0]), "points")
    if len(points[nom][0]) == 0:
        print("   -> ATTENTION : aucun point dans ce fichier, refais la capture")
print("Moteur d'affichage :", plt.get_backend())

# ---------- Création de la figure ----------
fig, ax = plt.subplots(figsize=(9, 9))
couleurs = plt.cm.tab10.colors
nuages, robots = {}, {}

for i, nom in enumerate(noms):
    coul = couleurs[i % 10]
    nuages[nom] = ax.scatter([], [], s=3, color=coul, label=f"{i + 1} : {nom}")
    robots[nom], = ax.plot([], [], "-o", color=coul, markevery=[0])  # position + direction du robot

selection = noms[0]


def redessiner():
    """Met à jour l'affichage de tous les scans."""
    for nom in noms:
        pose = SCANS[nom]
        nuages[nom].set_offsets(placer(*points[nom], pose))
        x0, y0, theta = pose
        dx, dy = 0.3 * math.cos(math.radians(theta)), 0.3 * math.sin(math.radians(theta))
        robots[nom].set_data([x0, x0 + dx], [y0, y0 + dy])
    x0, y0, theta = SCANS[selection]
    ax.set_title(f"Scan choisi : {selection}   x = {x0:.2f} m   y = {y0:.2f} m   θ = {theta:.0f}°")
    fig.canvas.draw_idle()


def touche(event):
    """Réagit aux touches du clavier."""
    global selection
    pose = SCANS[selection]
    if event.key and event.key.isdigit() and 1 <= int(event.key) <= len(noms):
        selection = noms[int(event.key) - 1]
    elif event.key == "left":
        pose[0] -= PAS_XY
    elif event.key == "right":
        pose[0] += PAS_XY
    elif event.key == "up":
        pose[1] += PAS_XY
    elif event.key == "down":
        pose[1] -= PAS_XY
    elif event.key == "a":
        pose[2] += PAS_THETA
    elif event.key == "z":
        pose[2] -= PAS_THETA
    redessiner()


fig.canvas.mpl_connect("key_press_event", touche)

# Premier affichage, puis on ajuste le cadre aux points (il ne bougera plus ensuite)
redessiner()
tous = [p for nom in noms for p in placer(*points[nom], SCANS[nom])]
ax.update_datalim(tous + [(x0, y0) for x0, y0, _ in SCANS.values()])
ax.autoscale_view()
ax.set_aspect("equal")
ax.margins(0.2)
ax.grid(True)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend(loc="upper right", markerscale=4)
plt.show()

# ---------- À la fermeture : poses finales à recopier pour la suite ----------
for nom in noms:
    x0, y0, theta = SCANS[nom]
    print(f'"{nom}": [{x0:.2f}, {y0:.2f}, {theta:.1f}],')