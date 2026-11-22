# robot_slam.py - Jour 3 : SLAM « j'avance, je m'arrête, je scanne » avec le robot piloté en WiFi
# Sur l'ESP32 : robot_slam_mvt.ino (il crée le WiFi « Robot_SLAM », mot de passe eigsi2026).
#
# Utilisation :
#   1. Allumer le robot, connecter le PC au WiFi Robot_SLAM, lancer ce script, CLIQUER dans la fenêtre.
#   2. Espace : 1er scan = scan de RÉFÉRENCE, il fixe le repère de la carte (pose 0, 0, 0).
#   3. Piloter le robot (garder z / q / s / d / a / e appuyée), relâcher : il s'arrête.
#   4. Espace : le robot envoie NB_TOURS tours du lidar (même fichier que capture.py), lire_scan (grille.py)
#      les filtre, le scan est localisé sur la carte déjà construite (multi-résolution de
#      localisation_grid.py, autour de la pose du scan précédent), puis ajouté à la grid map (grille.py).
#   5. Recommencer 3 et 4 dans toute la salle.
#   Bonus : o = mode automatique (le robot avance ou tourne seul d'après le dernier scan, puis scanne).
#
# Tout est enregistré dans robot_slam_sessions/session_.../ : scan_XX.csv, poses.csv, grille.png, grille.npy.
# Ce dossier se relit avec les outils des jours 1 et 2 (affichage.py pour retoucher les poses à la main,
# carte_vect.py pour la carte vectorielle...) : y copier ces scripts et les lancer.

import math
import os
import re
import socket
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# grille.py (jour 1) et localisation_grid.py (jour 2) : dans le même dossier que ce script, ou celui au-dessus
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path += [ICI, os.path.dirname(ICI)]
import grille as G               # lire_scan, nouvelle_carte, mettre_a_jour, boucher_trous, charger_poses
import localisation_grid as L    # reference_grille, multi_resolution, score, transformer

# ---------- Réglages ----------
IP, PORT = "192.168.4.1", 3333   # adresse du robot sur son propre réseau WiFi
NB_TOURS = 5                     # tours du lidar par scan (comme capture.py)
NB_POINTS = 400                  # sous-échantillonnage du scan à localiser (comme localisation_grid.py)
RAYON_RECHERCHE = 2.0            # on cherche le robot à moins de 2 m de la pose du scan précédent :
                                 # ne pas le déplacer de plus de ~1,5 m entre deux scans
SCORE_MIN = 0.40                 # en dessous, localisation douteuse : on demande avant d'ajouter le scan
DOSSIER = os.path.join(ICI, "robot_slam_sessions")
SESSION = None                   # None = nouvelle session ; "session_AAAAMMJJ_HHMMSS" = la reprendre

PERIODE = 100                    # ms : tant qu'une touche est appuyée, on renvoie l'ordre
                                 # (le robot s'arrête tout seul après 300 ms sans ordre)

# Bonus : mode automatique
DISTANCE_LIBRE = 0.80            # place libre devant le robot pour avancer (m)
DUREE_AVANCE = 1.0               # durée d'avance entre deux scans (s)
DUREE_TOURNE = 0.6               # durée de rotation quand c'est bloqué devant (s)

MOUVEMENTS = {"z": "avance", "s": "recule", "q": "tourne à gauche", "d": "tourne à droite",
              "a": "glisse à gauche", "e": "glisse à droite"}

AIDE = """PILOTER (garder la touche appuyée)
  z / s       avancer / reculer
  q / d       tourner à gauche / à droite
  a / e       glisser à gauche / à droite
  p / m       vitesse + / -

CARTOGRAPHIER (robot arrêté)
  Espace      scanner + localiser + ajouter
  Entrée / x  garder / jeter un scan douteux
  Retour arr. retirer le dernier scan
  o           mode automatique (bonus)
  c           (re)connecter le robot"""

# Matplotlib utilise déjà des touches (s = sauvegarder, q = quitter...) : on les libère toutes
for cle in list(plt.rcParams.keys()):
    if cle.startswith("keymap."):
        plt.rcParams[cle] = []


# ---------- Liaison WiFi avec le robot ----------
robot = None


def connecter():
    global robot
    if robot:
        robot.close()
    try:
        robot = socket.create_connection((IP, PORT), timeout=3)
        robot.settimeout(10)
        message("Connecté au robot.")
    except OSError:
        robot = None
        message(f"Robot non joignable ({IP}:{PORT}). Connecte le PC au WiFi Robot_SLAM, puis appuie sur c.")


def deconnecter(raison):
    global robot
    if robot:
        robot.close()
    robot = None
    message(raison + " Appuie sur c pour te reconnecter.")


def envoyer(texte):
    if robot is None:
        return False
    try:
        robot.sendall(texte.encode())
        return True
    except OSError:
        deconnecter("Connexion perdue.")
        return False


def demander_scan(fichier):
    """Demande NB_TOURS tours au robot et les écrit dans fichier, au format de capture.py.
    Renvoie (nb de points reçus, ligne de diagnostic du robot)."""
    if not envoyer(f"S{NB_TOURS}"):
        return 0, "robot non connecté (c pour le connecter)"
    donnees = b""
    try:
        while not donnees.endswith(b"FIN\n"):
            bloc = robot.recv(65536)
            if not bloc:
                raise OSError("connexion fermée par le robot")
            donnees += bloc
    except OSError as erreur:
        # on coupe : sinon la fin de ce scan arriverait au début du suivant
        deconnecter(f"Scan non reçu ({erreur}).")
        return 0, "scan non reçu"
    lignes = donnees.decode(errors="ignore").splitlines()
    points_lidar = [l for l in lignes if l.count(",") == 3]
    info = next((l[5:] for l in lignes if l.startswith("INFO ")), "")
    if points_lidar:
        with open(fichier, "w") as f:
            f.write("tour,angle_deg,distance_mm,intensite\n")
            f.write("\n".join(points_lidar) + "\n")
    return len(points_lidar), info


# ---------- Outils ----------
def points(scan):
    """(angle, distance) de lire_scan -> points (x, y) dans le repère du robot, sous-échantillonnés."""
    tous = np.array([(r * math.cos(a), r * math.sin(a)) for a, r in scan])
    return tous[np.linspace(0, len(tous) - 1, min(NB_POINTS, len(tous))).astype(int)]


def fleche(pose, taille=0.30):
    """Triangle qui représente le robot, pointe vers l'avant."""
    x, y, th = pose
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    forme = [(taille, 0), (-0.6 * taille, 0.45 * taille), (-0.3 * taille, 0), (-0.6 * taille, -0.45 * taille)]
    return [(x + c * u - s * v, y + s * u + c * v) for u, v in forme]


# ---------- SLAM ----------
scans = []       # scans gardés, dans l'ordre : {"nom", "scan" (lire_scan), "pose" [x, y, θ], "score"}
cartes = []      # cartes[k] = grid map avec les scans 0 à k (retirer le dernier scan = cartes.pop())
attente = None   # scan localisé avec un score trop faible : on attend Entrée (garder) ou x (jeter)
etat = {"touche": None, "appui": 0.0, "vitesse": 150, "occupe": False,
        "auto": False, "auto_touche": None, "auto_fin": 0.0, "message": ""}


def message(texte):
    etat["message"] = texte
    print(texte)


def localiser(carte, scan, autour):
    """Localise le scan sur la carte (jour 2), à moins de RAYON_RECHERCHE de la pose autour.
    Multi-résolution : toutes les orientations sont testées, pas besoin d'odométrie. Renvoie (pose, score)."""
    ref_haute = L.reference_grille(carte)                           # cellules de 5 cm
    ref_basse = L.reference_grille(carte, facteur=4, d_max=1.0)     # cellules de 20 cm
    x, y = autour[0], autour[1]
    zone = ((x - RAYON_RECHERCHE, x + RAYON_RECHERCHE), (y - RAYON_RECHERCHE, y + RAYON_RECHERCHE))
    pts = points(scan)
    x, y, th = L.multi_resolution(ref_basse, ref_haute, pts, zone)
    score = L.score(ref_haute, pts, x, y, th)
    return [float(x), float(y), float((th + 180) % 360 - 180)], float(score)


def ajouter(s):
    """Ajoute le scan s à la carte (grille.py) et enregistre la session."""
    carte = cartes[-1].copy() if cartes else G.nouvelle_carte()
    G.mettre_a_jour(carte, s["scan"], s["pose"])
    scans.append(s)
    cartes.append(carte)
    sauvegarder()


def sauvegarder():
    """poses.csv (lisible par grille.py, affichage.py, carte_vect.py) + la carte en image et en .npy."""
    with open(os.path.join(dossier, "poses.csv"), "w") as f:
        f.write("scan,x,y,theta\n")
        for s in scans:
            x, y, th = s["pose"]
            f.write(f"{s['nom']},{x:.3f},{y:.3f},{th:.2f}\n")
    if cartes:
        np.save(os.path.join(dossier, "grille.npy"), cartes[-1])
        image = 255 - G.boucher_trous(cartes[-1].copy())        # noir = occupé, gris = inconnu, blanc = libre
        plt.imsave(os.path.join(dossier, "grille.png"), image[::-1], cmap="gray", vmin=0, vmax=255)


def nom_suivant():
    """scan_XX.csv avec un numéro jamais utilisé dans le dossier (même par un scan rejeté)."""
    numeros = [int(n) for n in re.findall(r"scan_(\d+)\.csv", " ".join(os.listdir(dossier)))]
    return f"scan_{max(numeros, default=0) + 1:02d}.csv"


def mettre_de_cote(nom):
    """Scan rejeté : renommé rejete_scan_XX.csv, pour que les outils des jours 1 et 2 (scan_*.csv) l'ignorent."""
    fichier = os.path.join(dossier, nom)
    if os.path.exists(fichier):
        os.replace(fichier, os.path.join(dossier, "rejete_" + nom))


def scanner():
    """Scan + localisation + ajout à la carte. Renvoie True si le scan a été ajouté à la carte."""
    global attente
    if attente:
        message("Décide d'abord pour le scan en attente : Entrée = le garder, x = le jeter.")
        dessiner_etat()
        return False
    arreter()
    etat["occupe"] = True                        # pendant le scan, on ignore les touches et le minuteur
    try:
        message("Scan en cours : ne touche pas le robot...")
        rafraichir()
        nom = nom_suivant()
        fichier = os.path.join(dossier, nom)
        nb, info = demander_scan(fichier)
        if nb == 0:
            if robot:
                message(f"Aucun point reçu du lidar ({info}). Regarde le moniteur série de l'ESP32.")
            return False
        scan = G.lire_scan(fichier)
        if len(scan) < 50:
            mettre_de_cote(nom)
            message(f"{nom} : {len(scan)} points après lire_scan ({nb} reçus, {info}). Lidar masqué ou arrêté ?")
            return False
        s = {"nom": nom, "scan": scan, "pose": [0.0, 0.0, 0.0], "score": None}
        if not scans:
            ajouter(s)
            message(f"{nom} = scan de référence : il fixe le repère (0, 0, 0). Déplace le robot puis Espace.")
            return True
        message(f"{nom} reçu ({len(scan)} points). Localisation sur la carte...")
        rafraichir()
        t = time.time()
        s["pose"], s["score"] = localiser(cartes[-1], scan, scans[-1]["pose"])
        x, y, th = s["pose"]
        texte = (f"{nom} localisé en {time.time() - t:.1f} s : x = {x:.2f} m, y = {y:.2f} m, "
                 f"θ = {th:.0f}°, score {s['score']:.2f}.")
        if s["score"] >= SCORE_MIN:
            ajouter(s)
            message(texte + " Ajouté à la carte.")
            return True
        attente = s
        message(texte + " Score faible : les points violets collent-ils aux murs ? Entrée = garder, x = jeter.")
        return False
    finally:
        etat["occupe"] = False
        dessiner()


def garder_attente():
    global attente
    if attente:
        s, attente = attente, None
        ajouter(s)
        message(f"{s['nom']} gardé et ajouté à la carte.")
        dessiner()


def jeter_attente():
    global attente
    if attente:
        mettre_de_cote(attente["nom"])
        message(f"{attente['nom']} jeté. Ramène le robot vers une zone déjà cartographiée et rescanne.")
        attente = None
        dessiner()


def retirer_dernier():
    if attente:
        return jeter_attente()
    if scans:
        s = scans.pop()
        cartes.pop()
        mettre_de_cote(s["nom"])
        sauvegarder()
        message(f"{s['nom']} retiré de la carte.")
        dessiner()


# ---------- Bonus : mode automatique ----------
def degage(scan, a_min, a_max):
    """Distance de l'obstacle le plus proche entre a_min et a_max (°, 0 = devant, + = à gauche)."""
    d = [r for a, r in scan if a_min <= math.degrees(math.atan2(math.sin(a), math.cos(a))) <= a_max]
    return min(d, default=99.0)


def mouvement_auto():
    """D'après le dernier scan : avance si c'est libre devant, sinon tourne du côté le plus dégagé."""
    scan = scans[-1]["scan"]
    if degage(scan, -25, 25) > DISTANCE_LIBRE:
        return "z", DUREE_AVANCE
    return ("q" if degage(scan, 25, 90) > degage(scan, -90, -25) else "d"), DUREE_TOURNE


def basculer_auto():
    if etat["auto"]:
        etat["auto"] = False
        arreter()
        message("Mode automatique arrêté.")
    elif not scans:
        message("Fais d'abord le scan de référence (Espace).")
    elif attente:
        message("Décide d'abord pour le scan en attente : Entrée = le garder, x = le jeter.")
    else:
        etat["auto"] = True
        etat["auto_touche"], duree = mouvement_auto()
        etat["auto_fin"] = time.time() + duree
        message("Mode automatique : le robot avance, s'arrête, scanne... (o ou une touche de pilotage = stop)")
    dessiner_etat()


# ---------- Fenêtre ----------
fig = plt.figure("Robot SLAM - jour 3", figsize=(15, 9))
ax = fig.add_axes([0.04, 0.05, 0.60, 0.92])
D = G.TAILLE / 2
image = ax.imshow(np.full((G.N, G.N), 128), cmap="gray", vmin=0, vmax=255, origin="lower", extent=[-D, D, -D, D])
chemin, = ax.plot([], [], "o-", color="tab:blue", markersize=5, zorder=4, label="poses des scans")
nuage = ax.scatter([], [], s=3, color="red", zorder=3, label="dernier scan")
fleche_robot = Polygon(np.zeros((4, 2)), closed=True, color="darkorange", zorder=6, label="robot")
ax.add_patch(fleche_robot)
numeros = []
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend(loc="upper left", fontsize=8)
fig.text(0.67, 0.97, AIDE, va="top", family="monospace", fontsize=9)
texte_etat = fig.text(0.67, 0.58, "", va="top", family="monospace", fontsize=9)


def dessiner_etat():
    """Texte d'état (appelé souvent)."""
    if etat["auto"]:
        mouv = MOUVEMENTS.get(etat["auto_touche"], "scan") + " (auto)"
    else:
        mouv = MOUVEMENTS.get(etat["touche"], "arrêt")
    lignes = [f"ÉTAT",
              f"  Robot     {'connecté (' + IP + ')' if robot else 'NON CONNECTÉ (c)'}",
              f"  Vitesse   {etat['vitesse']}   (50 à 255)",
              f"  Mouvement {mouv}",
              f"  Session   {os.path.basename(dossier)}", ""]
    mots, ligne = etat["message"].split(), ""
    for mot in mots:                                  # message coupé en lignes de 55 caractères
        if len(ligne) + len(mot) > 55:
            lignes.append(ligne)
            ligne = ""
        ligne += mot + " "
    lignes += [ligne, "", "SCANS     x       y       θ    score"]
    for s in scans[::-1][:12]:
        x, y, th = s["pose"]
        score = "référence" if s["score"] is None else f"{s['score']:.2f}"
        lignes.append(f"  {s['nom'][5:-4]:>3s} {x:7.2f} {y:7.2f} {th:7.1f}°  {score}")
    texte_etat.set_text("\n".join(lignes))
    fig.canvas.draw_idle()


def dessiner():
    """Tout redessiner : carte, poses des scans, dernier scan (ou scan en attente), robot, cadrage."""
    carte = cartes[-1] if cartes else G.nouvelle_carte()
    image.set_data(255 - G.boucher_trous(carte.copy()))     # noir = occupé, gris = inconnu, blanc = libre
    P = np.array([s["pose"] for s in scans]).reshape(-1, 3)
    chemin.set_data(P[:, 0], P[:, 1])
    for t in numeros:
        t.remove()
    numeros.clear()
    for s in scans:
        numeros.append(ax.text(s["pose"][0] + 0.08, s["pose"][1] + 0.08, s["nom"][5:-4], fontsize=8,
                               color="tab:blue", zorder=5))
    dernier = attente or (scans[-1] if scans else None)
    if dernier:
        nuage.set_offsets(np.column_stack(L.transformer(points(dernier["scan"]), *dernier["pose"])))
        nuage.set_color("magenta" if attente else "red")
        fleche_robot.set_xy(fleche(dernier["pose"]))
    else:
        nuage.set_offsets(np.zeros((0, 2)))
    fleche_robot.set_visible(dernier is not None)
    lignes, colonnes = np.nonzero(carte != G.INCONNU)       # cadrage sur la partie connue de la carte
    if len(lignes):
        ax.set_xlim(-D + colonnes.min() * G.RESOLUTION - 0.5, -D + (colonnes.max() + 1) * G.RESOLUTION + 0.5)
        ax.set_ylim(-D + lignes.min() * G.RESOLUTION - 0.5, -D + (lignes.max() + 1) * G.RESOLUTION + 0.5)
    else:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    dessiner_etat()


def rafraichir():
    """Affiche tout de suite (pendant un scan, la fenêtre ne se redessine pas toute seule)."""
    dessiner_etat()
    fig.canvas.draw()
    fig.canvas.flush_events()


# ---------- Clavier, minuteur ----------
def arreter():
    etat["touche"] = None
    envoyer(" ")
    dessiner_etat()


def appui(event):
    if etat["occupe"]:
        return
    k = event.key or ""
    if k.lower() in MOUVEMENTS:                       # pilotage (majuscules acceptées)
        k = k.lower()
        if etat["auto"]:
            etat["auto"] = False
            message("Mode automatique arrêté.")
        etat["appui"] = time.time()
        if k != etat["touche"]:
            etat["touche"] = k
            envoyer(k)
            dessiner_etat()
    elif k in ("p", "m"):
        etat["vitesse"] = min(max(etat["vitesse"] + (25 if k == "p" else -25), 50), 255)
        envoyer(k)
        dessiner_etat()
    elif k == " " and not etat["auto"]:
        scanner()
    elif k == "enter":
        garder_attente()
    elif k == "x":
        jeter_attente()
    elif k == "backspace":
        retirer_dernier()
    elif k == "o":
        basculer_auto()
    elif k == "c":
        connecter()
        dessiner_etat()


def relache(event):
    if (event.key or "").lower() == etat["touche"]:
        arreter()


def tic():
    """Toutes les PERIODE ms : on renvoie l'ordre en cours (sinon le robot s'arrête) ; mode auto."""
    if etat["occupe"]:
        return
    if etat["auto"]:
        if time.time() < etat["auto_fin"]:            # mouvement automatique en cours
            envoyer(etat["auto_touche"])
            return
        etat["auto_touche"] = None                    # mouvement fini : on scanne
        if not scanner():
            etat["auto"] = False
            message(etat["message"] + " -> Mode automatique arrêté.")
            dessiner_etat()
            return
        etat["auto_touche"], duree = mouvement_auto()
        etat["auto_fin"] = time.time() + duree
        dessiner_etat()
        return
    k = etat["touche"]
    if k:
        if time.time() - etat["appui"] > 0.8 or not envoyer(k):   # touche plus vue (fenêtre quittée...)
            arreter()


def fermeture(event):
    envoyer(" ")
    if robot:
        robot.close()
    sauvegarder()


# ---------- Démarrage ----------
if SESSION:                                           # reprise d'une session enregistrée
    dossier = os.path.join(DOSSIER, SESSION)
    for nom, pose in G.charger_poses(os.path.join(dossier, "poses.csv")).items():
        ajouter({"nom": nom, "scan": G.lire_scan(os.path.join(dossier, nom)), "pose": list(pose), "score": None})
    message(f"Session reprise : {len(scans)} scans.")
else:
    dossier = os.path.join(DOSSIER, time.strftime("session_%Y%m%d_%H%M%S"))
    os.makedirs(dossier, exist_ok=True)
connecter()

fig.canvas.mpl_connect("key_press_event", appui)
fig.canvas.mpl_connect("key_release_event", relache)
fig.canvas.mpl_connect("close_event", fermeture)
minuteur = fig.canvas.new_timer(interval=PERIODE)
minuteur.add_callback(tic)
minuteur.start()
dessiner()

if __name__ == "__main__":
    plt.show()