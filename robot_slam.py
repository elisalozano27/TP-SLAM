# robot_slam.py - Jour 3 : SLAM en direct avec le robot piloté en WiFi (robot_slam_esp32.ino sur l'ESP32)
# 1. Allumer le robot : il crée le réseau WiFi « Robot_SLAM ». 2. Connecter le PC à ce réseau.
# 3. Lancer ce script, cliquer dans la fenêtre, piloter avec z q s d, scanner avec Espace.
#
# À chaque scan, le robot s'arrête et envoie NB_TOURS tours du lidar :
#   - 1er scan : il fixe le repère (pose 0, 0, 0) et crée la grid map (grille.py) ;
#   - scans suivants : calage sur la carte déjà construite (multi-résolution de localisation_grid.py),
#     en cherchant à moins de RAYON_RECHERCHE de la pose estimée du robot, puis ajout à la carte.
# Entre deux scans, la flèche avance selon les touches appuyées : estimation grossière (« odométrie »
# au temps), à étalonner avec V_AVANCE, V_LATERAL et V_ROTATION. Elle est recalée à chaque scan.
# Chaque scan est enregistré dans un dossier de session (même format que capture.py), avec poses.csv
# et l'image de la carte. Pour reprendre une session : SESSION = "nom du dossier".

import math
import os
import re
import socket
import sys
import textwrap
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button

# grille.py et localisation_grid.py (jours 1 et 2) : dans le même dossier que ce script, ou dans le dossier au-dessus
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path += [ICI, os.path.dirname(ICI)]
import grille as G
import localisation_grid as L

IP, PORT = "192.168.4.1", 3333   # adresse du robot sur son propre réseau (comme pilotage.py)
NB_TOURS = 4                     # tours du lidar par scan (la médiane de lire_scan veut 3 mesures à ±0,5° :
                                 # avec 2 tours, beaucoup de points seraient rejetés)
DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_slam_sessions")   # une session = un dossier
SESSION = None                   # None : nouvelle session ; sinon nom du dossier d'une session à reprendre

PERIODE = 100            # ms : tant qu'une touche est appuyée, on renvoie l'ordre (le robot s'arrête seul après 300 ms)
NB_POINTS = 400          # sous-échantillonnage du scan à caler
RAYON_RECHERCHE = 1.5    # le calage cherche le robot à moins de RAYON_RECHERCHE de sa pose estimée (m)
SCORE_MIN = 0.40         # en dessous, calage jugé douteux (croix rouge sur la carte)
VUE = 3.0                # demi-largeur de la vue « suivre le robot » (m)
PAS_XY, PAS_THETA = 0.02, 1.0   # correction à la main du scan choisi (m, °)

# Odométrie au temps : vitesses du robot à la vitesse VITESSE_DEFAUT (à mesurer : avancer 2 s, mesurer...)
V_AVANCE = 0.20          # m/s
V_LATERAL = 0.15         # m/s (translation)
V_ROTATION = 60.0        # °/s
VITESSE_DEFAUT, PAS_VITESSE, VITESSE_MIN, VITESSE_MAX = 150, 25, 50, 255   # comme dans l'ESP32

# touche : (avant, gauche, rotation, texte)
MOUVEMENTS = {"z": (1, 0, 0, "avance"), "s": (-1, 0, 0, "recule"), "q": (0, 0, 1, "tourne à gauche"),
              "d": (0, 0, -1, "tourne à droite"), "a": (0, 1, 0, "glisse à gauche"), "e": (0, -1, 0, "glisse à droite")}

AIDE = """PILOTER (garder la touche appuyée)
  z / s      avancer / reculer
  q / d      tourner à gauche / à droite
  a / e      glisser à gauche / à droite
  p / m      vitesse + / -

SCANNER
  Espace     scan + calage + carte

CORRIGER LA CARTE
  clic, n / b   choisir un scan
  flèches       déplacer le scan choisi
  r / t         le tourner (+ / -)
  c             le recaler automatiquement
  Retour arr.   supprimer le dernier scan
  Suppr         supprimer le scan choisi

VUE
  f          suivre le robot / carte entière"""

# Matplotlib utilise déjà des touches (s = sauvegarder, q = quitter...) : on les libère toutes
for cle in list(plt.rcParams.keys()):
    if cle.startswith("keymap."):
        plt.rcParams[cle] = []


# ---------- Outils ----------
def composer(ref, rel):
    """Pose dans la carte à partir de la pose ref et d'une pose rel exprimée dans le repère de ref."""
    c, s = math.cos(math.radians(ref[2])), math.sin(math.radians(ref[2]))
    return [ref[0] + c * rel[0] - s * rel[1], ref[1] + s * rel[0] + c * rel[1], ref[2] + rel[2]]


def relative(ref, pose):
    """Inverse de composer : pose exprimée dans le repère de ref."""
    dx, dy = pose[0] - ref[0], pose[1] - ref[1]
    c, s = math.cos(math.radians(ref[2])), math.sin(math.radians(ref[2]))
    return [c * dx + s * dy, -s * dx + c * dy, pose[2] - ref[2]]


def points(scan):
    """(angle, distance) de lire_scan -> points (x, y) dans le repère du robot, sous-échantillonnés."""
    tous = np.array([(r * math.cos(a), r * math.sin(a)) for a, r in scan])
    return tous[np.linspace(0, len(tous) - 1, min(NB_POINTS, len(tous))).astype(int)]


def fleche(pose, taille=0.30):
    """Triangle qui représente le robot, pointe vers l'avant."""
    forme = [(taille, 0), (-0.6 * taille, 0.45 * taille), (-0.3 * taille, 0), (-0.6 * taille, -0.45 * taille)]
    return [composer(pose, [x, y, 0])[:2] for x, y in forme]


# ---------- Liaison WiFi avec le robot ----------
robot = None


def connecter():
    global robot
    try:
        robot = socket.create_connection((IP, PORT), timeout=3)
        robot.settimeout(5)
        message("Connecté au robot.")
    except OSError as erreur:
        robot = None
        message(f"Robot non joignable ({IP}:{PORT}) : {erreur}. Connecte le PC au WiFi Robot_SLAM, puis « Reconnecter ».")


def envoyer(texte):
    global robot
    if robot is None:
        return False
    try:
        robot.sendall(texte.encode())
        return True
    except OSError:
        robot = None
        message("Connexion perdue : « Reconnecter ».")
        return False


def recevoir_scan(fichier):
    """Demande un scan au robot et l'enregistre dans fichier (format de capture.py). Renvoie le nb de points."""
    if not envoyer(f"S{NB_TOURS}"):
        message("Robot non connecté : impossible de scanner.")
        return 0
    donnees = b""
    try:
        while not donnees.endswith(b"FIN\n"):
            bloc = robot.recv(65536)
            if not bloc:
                raise OSError
            donnees += bloc
    except OSError:
        message("Scan non reçu (délai dépassé ou connexion coupée).")
        return 0
    lignes = [l for l in donnees.decode(errors="ignore").splitlines() if l.count(",") == 3]
    if len(lignes) < 50:
        message("Scan vide : le lidar tourne-t-il ?")
        return 0
    with open(fichier, "w") as f:
        f.write("tour,angle_deg,distance_mm,intensite\n")
        f.write("".join(l + "\n" for l in lignes))
    return len(lignes)


# ---------- SLAM ----------
scans = []      # un dict par scan : nom du fichier, scan (lire_scan), pose [x, y, θ], score du calage
cartes = []     # cartes[k] = grid map avec les scans 0 à k (on garde tout : retour en arrière rapide)
trace = []      # poses estimées du robot depuis le dernier scan (chemin parcouru)
etat = {"pose": [0.0, 0.0, 0.0], "touche": None, "appui": 0.0, "t": time.time(), "vitesse": VITESSE_DEFAUT,
        "choisi": None, "suivre": False, "message": ""}


def message(texte):
    etat["message"] = texte
    print(texte)


def carte_actuelle():
    return cartes[-1] if cartes else G.nouvelle_carte()


def localiser(carte, scan, estimation):
    """Cale le scan sur la carte, autour de la pose estimée. Renvoie (pose, score)."""
    ref_haute = L.reference_grille(carte)                           # cellules de 5 cm
    ref_basse = L.reference_grille(carte, facteur=4, d_max=1.0)     # cellules de 20 cm (§8.2)
    x, y, _ = estimation
    zone = ((x - RAYON_RECHERCHE, x + RAYON_RECHERCHE), (y - RAYON_RECHERCHE, y + RAYON_RECHERCHE))
    pts = points(scan)
    pose = L.multi_resolution(ref_basse, ref_haute, pts, zone)      # toutes les orientations sont testées
    return pose, L.score(ref_haute, pts, *pose)


def reconstruire(depuis):
    """Recalcule les cartes à partir du scan n° depuis (les cartes d'avant ne changent pas)."""
    carte = cartes[depuis - 1].copy() if depuis > 0 else G.nouvelle_carte()
    del cartes[depuis:]
    for s in scans[depuis:]:
        G.mettre_a_jour(carte, s["scan"], s["pose"])
        cartes.append(carte.copy())


def sauvegarder():
    """poses.csv (lisible par grille.py) + image de la carte, dans le dossier de la session."""
    with open(os.path.join(dossier, "poses.csv"), "w") as f:
        f.write("scan,x,y,theta,score\n")
        for s in scans:
            x, y, th = s["pose"]
            score = "" if s["score"] is None else f"{s['score']:.3f}"
            f.write(f"{s['nom']},{x:.3f},{y:.3f},{th:.2f},{score}\n")
    image = np.where(G.boucher_trous(carte_actuelle().copy()) > G.INCONNU, 0, 255)   # noir = occupé
    plt.imsave(os.path.join(dossier, "grille.png"), image[::-1], cmap="gray", vmin=0, vmax=255)


def nouveau_scan():
    etat["touche"] = None
    message("Scan en cours, ne bouge pas le robot...")
    dessiner()
    fig.canvas.flush_events()
    numeros = [int(n) for n in re.findall(r"scan_(\d+)\.csv", " ".join(os.listdir(dossier)))]
    nom = f"scan_{max(numeros, default=0) + 1:02d}.csv"          # jamais un numéro déjà utilisé
    if recevoir_scan(os.path.join(dossier, nom)) == 0:
        return dessiner()
    scan = G.lire_scan(os.path.join(dossier, nom))
    if len(scan) < 50:
        message(f"{nom} : presque aucun point après les filtres de lire_scan (augmente NB_TOURS ?).")
        return dessiner()
    if not scans:
        pose, score = list(etat["pose"]), None                     # 1er scan : il fixe le repère
        texte = f"{nom} : 1er scan, il fixe le repère."
    else:
        t = time.time()
        pose, score = localiser(cartes[-1], scan, etat["pose"])
        texte = f"{nom} calé en {time.time() - t:.1f} s, score {score:.2f}."
        if score < SCORE_MIN:
            texte += " CALAGE DOUTEUX : corrige-le (flèches, r / t, c) ou supprime-le (Retour arrière)."
    scans.append({"nom": nom, "scan": scan, "pose": pose, "score": score})
    reconstruire(len(scans) - 1)
    etat["pose"], etat["choisi"] = list(pose), len(scans) - 1
    trace.clear()
    message(texte)
    sauvegarder()
    dessiner()


def placer(i, pose):
    """Change la pose du scan i et recalcule la carte. Si c'est le dernier scan, la flèche suit."""
    ancienne = scans[i]["pose"]
    scans[i]["pose"] = list(pose)
    if i == len(scans) - 1:
        etat["pose"] = composer(pose, relative(ancienne, etat["pose"]))
        trace[:] = [composer(pose, relative(ancienne, p)) for p in trace]
    reconstruire(i)
    sauvegarder()
    dessiner()


def corriger(dx, dy, dtheta):
    i = etat["choisi"]
    if i is not None:
        x, y, th = scans[i]["pose"]
        placer(i, [x + dx, y + dy, th + dtheta])


def recaler():
    i = etat["choisi"]
    if not i:                                    # None ou 0
        message("Le 1er scan sert de repère : rien à recaler." if i == 0 else "Aucun scan choisi.")
        return dessiner()
    pose, score = localiser(cartes[i - 1], scans[i]["scan"], scans[i]["pose"])   # carte sans ce scan ni les suivants
    scans[i]["score"] = score
    message(f"{scans[i]['nom']} recalé, score {score:.2f}.")
    placer(i, pose)


def supprimer(i):
    if i is None or not scans:
        return
    nom = scans.pop(i)["nom"]
    reconstruire(i)
    etat["choisi"] = min(i, len(scans) - 1) if scans else None
    message(f"{nom} retiré de la carte (son fichier reste dans le dossier).")
    sauvegarder()
    dessiner()


# ---------- Fenêtre ----------
fig = plt.figure("Robot SLAM - jour 3", figsize=(15, 9))
ax = fig.add_axes([0.04, 0.05, 0.58, 0.91])
d = G.TAILLE / 2
image = ax.imshow(np.full((G.N, G.N), 128), cmap="gray", vmin=0, vmax=255, origin="lower", extent=[-d, d, -d, d])
nuage = ax.scatter([], [], s=3, color="red", zorder=3, label="points du scan choisi")
chemin, = ax.plot([], [], "o-", color="tab:blue", markersize=5, zorder=4, label="poses des scans")
douteux, = ax.plot([], [], "x", color="red", markersize=12, mew=2, zorder=4, label="calage douteux")
cercle_choisi, = ax.plot([], [], "o", mfc="none", mec="gold", markersize=16, mew=2.5, zorder=4)
trajet, = ax.plot([], [], ":", color="darkorange", linewidth=2, zorder=4, label="trajet depuis le dernier scan")
fleche_robot = Polygon(fleche(etat["pose"]), closed=True, color="darkorange", zorder=6, label="robot")
ax.add_patch(fleche_robot)
numeros = []
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend(loc="upper left", fontsize=8)
fig.text(0.65, 0.97, AIDE, va="top", family="monospace", fontsize=9)
texte_etat = fig.text(0.65, 0.50, "", va="top", family="monospace", fontsize=9)


def texte_scans():
    lignes = ["SCANS   n°     x       y       θ    score"]
    for i in range(len(scans) - 1, max(len(scans) - 9, 0) - 1, -1):    # les 8 derniers, du plus récent
        s = scans[i]
        x, y, th = s["pose"]
        score = "repère" if s["score"] is None else f"{s['score']:.2f}" + (" !" if s["score"] < SCORE_MIN else "")
        lignes.append(f"  {'>' if i == etat['choisi'] else ' '}   {s['nom'][5:-4]:>3s} {x:7.2f} {y:7.2f} {th:7.1f}°  {score}")
    return "\n".join(lignes)


def dessiner_robot():
    """Flèche, trajet depuis le dernier scan et texte d'état (appelé souvent)."""
    fleche_robot.set_xy(fleche(etat["pose"]))
    T = np.array(([scans[-1]["pose"]] if scans else []) + trace + [etat["pose"]])
    trajet.set_data(T[:, 0], T[:, 1])
    x, y, th = etat["pose"]
    if etat["suivre"]:
        ax.set_xlim(x - VUE, x + VUE)
        ax.set_ylim(y - VUE, y + VUE)
    mouv = MOUVEMENTS[etat["touche"]][3] if etat["touche"] else "arrêt"
    texte_etat.set_text(
        f"ÉTAT\n"
        f"  Robot       {'connecté (' + IP + ')' if robot else 'NON CONNECTÉ'}\n"
        f"  Vitesse     {etat['vitesse']}   ({VITESSE_MIN} à {VITESSE_MAX})\n"
        f"  Mouvement   {mouv}\n"
        f"  Pose robot  x = {x:.2f} m   y = {y:.2f} m   θ = {th:.0f}°\n"
        f"  Session     {dossier}\n\n"
        + "\n".join(textwrap.wrap(etat["message"], 52)) + "\n\n" + texte_scans())
    fig.canvas.draw_idle()


def dessiner():
    """Tout redessiner : carte, poses des scans, scan choisi, cadrage (après chaque changement de carte)."""
    carte = carte_actuelle()
    image.set_data(255 - G.boucher_trous(carte.copy()))          # noir = occupé, gris = inconnu, blanc = libre
    P = np.array([s["pose"] for s in scans]).reshape(-1, 3)
    chemin.set_data(P[:, 0], P[:, 1])
    D = np.array([s["pose"] for s in scans if s["score"] is not None and s["score"] < SCORE_MIN]).reshape(-1, 3)
    douteux.set_data(D[:, 0], D[:, 1])
    for t in numeros:
        t.remove()
    numeros.clear()
    for s in scans:
        numeros.append(ax.text(s["pose"][0] + 0.08, s["pose"][1] + 0.08, s["nom"][5:-4], fontsize=8,
                               color="tab:blue", zorder=5))
    i = etat["choisi"]
    if i is None:
        nuage.set_offsets(np.zeros((0, 2)))
        cercle_choisi.set_data([], [])
    else:
        nuage.set_offsets(np.column_stack(L.transformer(points(scans[i]["scan"]), *scans[i]["pose"])))
        cercle_choisi.set_data([scans[i]["pose"][0]], [scans[i]["pose"][1]])
    if not etat["suivre"]:                                        # cadrage sur la partie connue de la carte
        lignes, colonnes = np.nonzero(carte != G.INCONNU)
        x, y = etat["pose"][:2]
        if len(lignes):
            xs = -d + np.array([colonnes.min(), colonnes.max() + 1]) * G.RESOLUTION
            ys = -d + np.array([lignes.min(), lignes.max() + 1]) * G.RESOLUTION
            ax.set_xlim(min(xs[0], x) - 0.5, max(xs[1], x) + 0.5)
            ax.set_ylim(min(ys[0], y) - 0.5, max(ys[1], y) + 0.5)
        else:
            ax.set_xlim(x - VUE, x + VUE)
            ax.set_ylim(y - VUE, y + VUE)
    dessiner_robot()


# ---------- Clavier, souris, boutons ----------
def appui(event):
    k = event.key or ""
    if k.lower() in MOUVEMENTS:                   # pilotage (majuscules acceptées)
        k = k.lower()
        etat["appui"] = time.time()
        if k != etat["touche"]:
            etat["touche"], etat["t"] = k, time.time()
            envoyer(k)
            dessiner_robot()
    elif k in ("p", "m"):
        etat["vitesse"] = min(max(etat["vitesse"] + (PAS_VITESSE if k == "p" else -PAS_VITESSE), VITESSE_MIN), VITESSE_MAX)
        envoyer(k)
        dessiner_robot()
    elif k == " ":
        nouveau_scan()
    elif k == "backspace":
        supprimer(len(scans) - 1 if scans else None)
    elif k == "delete":
        supprimer(etat["choisi"])
    elif k in ("n", "b") and scans:
        etat["choisi"] = ((etat["choisi"] or 0) + (1 if k == "n" else -1)) % len(scans)
        dessiner()
    elif k in ("left", "right", "up", "down"):
        corriger(*{"left": (-PAS_XY, 0), "right": (PAS_XY, 0), "up": (0, PAS_XY), "down": (0, -PAS_XY)}[k], 0)
    elif k in ("r", "t"):
        corriger(0, 0, PAS_THETA if k == "r" else -PAS_THETA)
    elif k == "c":
        recaler()
    elif k == "f":
        basculer_suivi()


def relache(event):
    if (event.key or "").lower() == etat["touche"]:
        arreter()


def arreter():
    etat["touche"] = None
    envoyer(" ")
    dessiner_robot()


def basculer_suivi():
    etat["suivre"] = not etat["suivre"]
    dessiner()


def clic(event):
    """Clic gauche près de la pose d'un scan : ce scan devient le scan choisi."""
    if event.inaxes is not ax or event.button != 1 or not scans or event.xdata is None:
        return
    dist = [math.dist((event.xdata, event.ydata), s["pose"][:2]) for s in scans]
    i = int(np.argmin(dist))
    if dist[i] < 0.03 * (ax.get_xlim()[1] - ax.get_xlim()[0]):
        etat["choisi"] = i
        dessiner()


def tic():
    """Toutes les PERIODE ms : on renvoie l'ordre en cours et on fait avancer la flèche."""
    maintenant = time.time()
    dt, etat["t"] = maintenant - etat["t"], maintenant
    k = etat["touche"]
    if k is None:
        return
    if maintenant - etat["appui"] > 0.8:          # plus de nouvelle d'une touche appuyée (fenêtre quittée...) : arrêt
        return arreter()
    if not envoyer(k):
        return arreter()
    avant, gauche, rotation, _ = MOUVEMENTS[k]
    f = etat["vitesse"] / VITESSE_DEFAUT
    x, y, th = etat["pose"]
    th += rotation * V_ROTATION * f * dt
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    x += (avant * V_AVANCE * c - gauche * V_LATERAL * s) * f * dt
    y += (avant * V_AVANCE * s + gauche * V_LATERAL * c) * f * dt
    etat["pose"] = [x, y, th]
    trace.append([x, y, th])
    dessiner_robot()


def fermeture(event):
    envoyer(" ")
    if robot:
        robot.close()
    sauvegarder()


boutons = []
for k, (texte, action) in enumerate([("Scanner (Espace)", nouveau_scan),
                                     ("Supprimer le dernier", lambda: supprimer(len(scans) - 1 if scans else None)),
                                     ("Supprimer le choisi", lambda: supprimer(etat["choisi"])),
                                     ("Recaler le choisi (c)", recaler),
                                     ("Suivre le robot (f)", basculer_suivi),
                                     ("Reconnecter", lambda: (connecter(), dessiner_robot()))]):
    b = Button(fig.add_axes([0.65 + (k % 3) * 0.115, 0.08 - (k // 3) * 0.055, 0.11, 0.045]), texte)
    b.on_clicked(lambda event, action=action: action())
    boutons.append(b)

# ---------- Démarrage ----------
if SESSION:                                       # reprise d'une session enregistrée
    dossier = os.path.join(DOSSIER, SESSION)
    for nom, pose in G.charger_poses(os.path.join(dossier, "poses.csv")).items():
        scans.append({"nom": nom, "scan": G.lire_scan(os.path.join(dossier, nom)), "pose": pose, "score": None})
    reconstruire(0)
    if scans:
        etat["pose"], etat["choisi"] = list(scans[-1]["pose"]), len(scans) - 1
else:
    dossier = os.path.join(DOSSIER, time.strftime("session_%Y%m%d_%H%M%S"))
    os.makedirs(dossier, exist_ok=True)
connecter()

fig.canvas.mpl_connect("key_press_event", appui)
fig.canvas.mpl_connect("key_release_event", relache)
fig.canvas.mpl_connect("button_press_event", clic)
fig.canvas.mpl_connect("close_event", fermeture)
minuteur = fig.canvas.new_timer(interval=PERIODE)
minuteur.add_callback(tic)
minuteur.start()
if __name__ == "__main__":
    dessiner()
    plt.show()