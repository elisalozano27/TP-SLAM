# TP SLAM – Lidar LD06

Cartographie d'une salle et localisation du robot MecanumCarV2 avec un lidar LD06 et un ESP32.
Le compte rendu complet (démarche, choix, résultats, limites) est dans `compte_rendu_slam_lidar.html` : il s'ouvre dans n'importe quel navigateur.

## Installation

**PC** : Python 3.10 ou plus récent, puis :

```
pip install numpy matplotlib pyserial
```

`pyserial` ne sert qu'à `capture.py`. Tous les programmes se lancent **depuis le dossier du dépôt**, car ils s'importent entre eux et lisent les fichiers de données qui s'y trouvent.

**Robot** : IDE Arduino avec le support des cartes ESP32 et la bibliothèque LD06 fournie dans les ressources du module. Câblage du lidar : Tx du lidar sur la broche 33 de l'ESP32, PWM sur la broche 32, masse commune.

## Contenu du dépôt

| Fichier | Rôle |
| --- | --- |
| `capture.py` | Enregistre des tours du lidar (USB) dans un fichier CSV |
| `grille.py` | Lecture et filtrage des scans (`lire_scan`), construction et mise à jour de la grid map |
| `affichage.py` | Affiche tous les scans et permet de régler leur pose à la main, enregistre `poses.csv` |
| `segment.py` | Nuage de points → segments, et opérations sur les segments (avec tests) |
| `carte_vect.py` | Construction de la carte vectorielle |
| `localisation_grid.py` | Les 6 méthodes de localisation ; localise un vrai scan sur la grid map des autres |
| `localisation_map_tool.py` | Comparaison des méthodes sur une grid map mapTool (pose connue) |
| `localisation_vect.py` | Comparaison des méthodes sur une carte vectorielle mapTool (pose connue) |
| `localisation_scan_simulation.py` | Localisation entre deux scans mapTool (poses connues) |
| `localisation_scan.py` | Localisation entre scans, de proche en proche, sur les vrais scans |
| `robot_slam.py` | SLAM avec le robot piloté en WiFi |
| `rejouer_slam.py` | Rejoue un dossier de scans dans la boucle SLAM, sans robot |
| `robot_slam_mvt.ino` | Programme de l'ESP32 pour le SLAM (WiFi, moteurs, scans à la demande) |
| `ESP32_LD06_Test/` | Programme de l'ESP32 pour les jours 1 et 2 (envoi des scans en USB pour `capture.py`) |
| `sens_moteurs/` | Test des broches des moteurs (quelle broche fait tourner quelle roue) |

**Données**

| Fichier | Contenu |
| --- | --- |
| `scan_01.csv` … `scan_09.csv` | Mes 9 scans réels de la salle, 5 tours chacun : `tour,angle_deg,distance_mm,intensite` |
| `poses.csv` | Pose de chaque scan, réglée à la main avec `affichage.py` : `scan,x,y,theta` (m, m, °) |
| `Map_Tool_grid_map.csv`, `Map_Tool_vect.txt`, `Map_Tool_scan_lidar.csv`, `Map_Tool_scan_lidar_21.csv` | Première scène mapTool : grid map, carte vectorielle, deux scans (pose vraie en 1re ligne) |
| `Map_Tool_grid_map_2.csv`, `Map_Tool_vect_2.txt`, `Map_Tool_scan_lidar_2.csv` | Seconde scène mapTool |
| `captures_102/` | 102 scans de la salle le long d'un trajet, avec leurs poses de référence (`poses_vraies_simulation.csv`), pour valider le SLAM (voir le compte rendu) |

## Tester les outils

Les réglages (fichier à traiter, seuils…) sont des constantes en majuscules au début de chaque programme.
Les temps indiqués sont des ordres de grandeur.

### Jour 1

**Capturer un scan** (robot branché en USB, `ESP32_LD06_Test.ino` téléversé)
```
python capture.py
```
Régler `PORT` (ex. `COM3`) et `FICHIER` au début du fichier. Le programme enregistre `NB_TOURS` tours dans le CSV.

**Afficher et aligner les scans**
```
python affichage.py
```
Choisir un scan dans la liste, le déplacer avec les curseurs. Clavier (cliquer d'abord dans la fenêtre) : flèches = ±1 cm, `a` / `z` = ±0,5°, `n` / `b` = scan suivant / précédent. Le bouton *Sauvegarder* écrit `poses.csv`.

**Grid map**
```
python grille.py
```
Affiche la carte créée avec le premier scan, puis la même carte mise à jour avec tous les scans de `poses.csv`.

**Segments**
```
python segment.py          # segments extraits de scan_01.csv
python segment.py test     # tests des fonctions et des cas limites : toutes les lignes doivent afficher OK
```

**Carte vectorielle**
```
python carte_vect.py
```
Affiche la carte (environ 33 segments, épaisseur = confiance) et l'enregistre dans `carte_vectorielle.txt`.

### Jour 2

**Localisation d'un vrai scan sur la grid map**
```
python localisation_grid.py
```
Construit la grid map sans `scan_05`, puis localise `scan_05` par multi-résolution (sans estimation initiale, environ 5 s) et par gradient (estimation fausse de 40 cm, 30 cm et 8°). Résultat attendu : environ 6 cm et 0,6° d'écart avec `poses.csv` pour les deux méthodes. Changer `SCAN_TEST` pour localiser un autre scan.

**Comparaison des méthodes sur mapTool (poses connues)**
```
python localisation_map_tool.py          # grid map, seconde scène (environ 20 s)
python localisation_vect.py              # carte vectorielle, première scène (environ 1 min)
python localisation_scan_simulation.py   # entre deux scans de la première scène (quelques secondes)
```
Chaque programme affiche les tableaux de résultats dans la console (erreurs, temps, taux de réussite) et les figures des poses trouvées. La vraie pose n'est lue qu'après les recherches. `localisation_vect.py` trace aussi le taux de réussite selon l'erreur de départ (test 3).

**Localisation entre les vrais scans, de proche en proche**
```
python localisation_scan.py
```
Affiche l'écart avec `poses.csv` pour chaque scan et les trajectoires obtenues. Les 9 scans étant espacés de plusieurs mètres, la dérive est importante : c'est analysé dans le compte rendu.

### Jour 3

**SLAM avec le robot**
1. Téléverser `robot_slam_mvt.ino` sur l'ESP32 et allumer le robot : il crée le réseau WiFi `Robot_SLAM` (mot de passe `eigsi2026`). Le moniteur série (115200 bauds) affiche l'état du lidar toutes les 2 s.
2. Connecter le PC à ce réseau, puis :
   ```
   python robot_slam.py
   ```
3. Cliquer dans la fenêtre. `Espace` = scan (le premier fixe le repère), `z` `s` `q` `d` `a` `e` = déplacer le robot (garder la touche appuyée), `p` / `m` = vitesse, `Entrée` / `x` = garder / jeter un scan douteux, `Retour arrière` = retirer le dernier scan, `o` = mode automatique, `c` = reconnecter.

Chaque session est enregistrée dans `robot_slam_sessions/`.

**Valider le SLAM sans robot**
```
python rejouer_slam.py captures_102
```
Rejoue les 102 scans dans la même boucle que `robot_slam.py` (environ 1 min), affiche la carte et compare chaque pose à sa pose de référence. Résultat attendu : erreur de position moyenne d'environ 3,7 cm (6,3 cm au maximum) et moins de 0,5° d'erreur d'angle. Les poses trouvées sont enregistrées dans `captures_102/poses_slam.csv`.

### Robot : tests matériels

- `sens_moteurs/sens_moteurs.ino` met chaque broche à l'état haut pendant 2 s : noter quelle roue tourne et dans quel sens, puis reporter les broches dans `ROUES` au début de `robot_slam_mvt.ino`.
- Si le moniteur série de `robot_slam_mvt.ino` affiche 0 deg/s et 0 erreur CRC, l'ESP32 ne reçoit rien du lidar : vérifier l'alimentation du lidar (robot allumé), le fil Tx sur la broche 33 et la masse commune.
