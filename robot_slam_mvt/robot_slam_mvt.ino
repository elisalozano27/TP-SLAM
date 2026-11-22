// robot_slam_mvt.ino - Jour 3 : pilotage du robot en WiFi + scans du lidar à la demande (pour robot_slam.py)
// L'ESP32 crée son propre réseau WiFi (SSID / MDP) et attend le PC sur le port PORT.
// Le PC envoie une lettre par ordre :
//   z avance, s recule, q tourne à gauche, d tourne à droite, a glisse à gauche, e glisse à droite,
//   espace = arrêt, p / m = plus vite / moins vite,
//   S suivi d'un chiffre (ex. "S5") = le robot s'arrête, ENREGISTRE 5 tours du lidar, puis les envoie.
// Réponse à un scan : une ligne par point "tour,angle_deg,distance_mm,intensite" (format de capture.py),
// puis une ligne "INFO ..." (diagnostic) et une ligne "FIN".
// Sécurité : sans nouvel ordre pendant TIMEOUT_MS (touche relâchée, WiFi coupé...), le robot s'arrête.
// Moniteur série (115200) : état du lidar toutes les 2 s, pour vérifier qu'il tourne et qu'il est bien lu.
//
// Ce qui change par rapport à la version qui ne scannait pas :
//  1. Les tours sont d'abord enregistrés en mémoire, PUIS envoyés. Avant, on envoyait un tour en WiFi
//     pendant que le lidar continuait d'envoyer ses données : son tampon série (4 Ko = 0,2 s) débordait,
//     les tours suivants étaient abîmés et jamais complets -> scan vide.
//  2. Si le PC se reconnecte (touche c), la nouvelle connexion remplace l'ancienne.
//  3. Serial1 est ouvert une seule fois sur la broche 33 (sans ld06.init()), comme test_lidar.ino.
// Le lidar est décodé avec la bibliothèque LD06 du prof (readScan, getPoints...).

#include <WiFi.h>
#include "ld06.h"

const char* SSID = "Robot_SLAM";     // nom du réseau créé par le robot
const char* MDP = "eigsi2026";       // mot de passe (8 caractères minimum)
const int PORT = 3333;
const uint32_t TIMEOUT_MS = 300;          // arrêt automatique sans ordre pendant ce temps
const uint32_t ATTENTE_SCAN_MS = 400;     // avant de mesurer, le robot reste arrêté ce temps (fin des secousses)
const uint32_t DUREE_MAX_SCAN_MS = 5000;  // abandon si les tours n'arrivent pas (lidar débranché...)

// Lidar : mêmes réglages que l'exemple du prof (bibliothèque LD06)
#define RX_LIDAR 33
#define TX_LIDAR -1
#define LD06_PWM_PIN 32
#define LIDAR_BAUD 230400
LD06 ld06(Serial1, LD06_PWM_PIN);

// Roues : {pin qui fait AVANCER le robot, pin qui le fait RECULER}, relevées avec test_pins.ino.
// Si une roue tourne à l'envers quand on appuie sur z : échanger ses deux pins ici.
const int ROUES[4][2] = {
  {19, 21},    // avant gauche
  {18, 5},     // avant droite
  {2, 4},      // arrière gauche
  {17, 16},    // arrière droite
};
WiFiServer serveur(PORT);
WiFiClient client;

char mouvement = ' ';                // mouvement en cours (' ' = arrêté)
uint32_t dernierOrdre = 0;           // instant du dernier ordre reçu (ms)
int vitesse = 150;                   // rapport cyclique PWM des roues (50 à 255)

// Points d'un scan, enregistrés en mémoire avant l'envoi.
// Le tableau est créé au démarrage avec malloc (dans le tas) : en variable globale il ne tient pas
// dans la RAM statique de l'ESP32 (erreur « dram0_0_seg overflowed » à la compilation).
const int MAX_MESURES = 6000;        // 5 tours x ~500 points (10 tours/s) = 2500 : large marge
struct Mesure {
  uint16_t angle100;                 // angle en centièmes de degré
  uint16_t distance;                 // mm
  uint8_t intensite;
  uint8_t tour;
};
Mesure* mesures = nullptr;
int maxMesures = 0;                  // nb de points que le tableau peut contenir
int nbMesures = 0;

uint32_t toursLus = 0;               // pour le diagnostic
uint16_t pointsDernierTour = 0;

bool lireLidar() {
  // A appeler très souvent : renvoie true quand un nouveau tour complet est disponible
  if (!ld06.readScan()) return false;
  toursLus++;
  pointsDernierTour = ld06.getNbPointsInScan();
  return true;
}

void roues(int ag, int ad, int arg, int ard) {
  // sens de chaque roue : +1 avance, -1 recule, 0 arrêt (avant gauche, avant droite, arrière gauche, arrière droite)
  int sens[4] = {ag, ad, arg, ard};
  for (int i = 0; i < 4; i++) {
    analogWrite(ROUES[i][0], sens[i] > 0 ? vitesse : 0);
    analogWrite(ROUES[i][1], sens[i] < 0 ? vitesse : 0);
  }
}

void executer(char c) {
  if (c == 'p' || c == 'm') {        // changement de vitesse
    vitesse = constrain(vitesse + (c == 'p' ? 25 : -25), 50, 255);
    char m = mouvement;              // on relance le mouvement en cours avec la nouvelle vitesse
    mouvement = ' ';
    if (m != ' ') executer(m);
    return;
  }
  if (c == mouvement) {              // même ordre : on prolonge seulement le mouvement
    dernierOrdre = millis();
    return;
  }
  switch (c) {
    case 'z': roues( 1,  1,  1,  1); break;   // avance
    case 's': roues(-1, -1, -1, -1); break;   // recule
    case 'q': roues(-1,  1, -1,  1); break;   // tourne à gauche
    case 'd': roues( 1, -1,  1, -1); break;   // tourne à droite
    case 'a': roues(-1,  1,  1, -1); break;   // glisse à gauche (roues mecanum)
    case 'e': roues( 1, -1, -1,  1); break;   // glisse à droite
    case ' ': roues( 0,  0,  0,  0); break;   // arrêt
    default: return;                          // caractère inconnu : ignoré
  }
  mouvement = c;
  dernierOrdre = millis();
}

// ---------- Scan ----------
// Le scan n'a PAS sa propre boucle de lecture : c'est loop() qui lit le lidar en continu (ça, ça marche),
// et pendant un scan il recopie les tours complets qu'il reçoit dans le tableau mesures.
int toursVoulus = 0;                 // > 0 : un scan est en cours
int tourScan = 0;                    // n° du tour en cours d'enregistrement (0 = tour jeté)
uint32_t debutScan = 0;
uint32_t toursLusDebut = 0;

void demarrerScan(int nbTours) {
  executer(' ');                     // on s'arrête
  mouvement = ' ';
  roues(0, 0, 0, 0);
  toursVoulus = nbTours;
  tourScan = 0;
  nbMesures = 0;
  debutScan = millis();
  toursLusDebut = toursLus;
}

void envoyerScan(int toursEnregistres);   // écrite juste après

void suivreScan(bool nouveauTour) {
  // Appelée à chaque loop() pendant un scan, avec le résultat de lireLidar()
  if (nouveauTour && millis() - debutScan > ATTENTE_SCAN_MS) {   // on attend la fin des secousses
    if (tourScan > 0) {              // le tour 0 a pu commencer avant l'arrêt complet : on le jette
      for (uint16_t i = 0; i < pointsDernierTour && nbMesures < maxMesures; i++) {
        DataPoint* p = ld06.getPoints(i);
        mesures[nbMesures++] = {(uint16_t)(p->angle * 100 + 0.5f), p->distance, p->intensity, (uint8_t)tourScan};
      }
    }
    tourScan++;
  }
  if (tourScan > toursVoulus || millis() - debutScan > DUREE_MAX_SCAN_MS) {
    envoyerScan(tourScan > 0 ? tourScan - 1 : 0);
    toursVoulus = 0;
  }
}

void envoyerScan(int toursEnregistres) {
  // Tout le scan d'un coup, par paquets d'environ 1,4 Ko (taille d'un paquet WiFi)
  char tampon[1460];
  size_t n = 0;
  for (int i = 0; i < nbMesures; i++) {
    const Mesure& m = mesures[i];
    n += snprintf(tampon + n, sizeof(tampon) - n, "%d,%d.%02d,%d,%d\n",
                  (int)m.tour, m.angle100 / 100, m.angle100 % 100, (int)m.distance, (int)m.intensite);
    if (n > sizeof(tampon) - 40) {
      client.write((const uint8_t*)tampon, n);
      n = 0;
    }
  }
  n += snprintf(tampon + n, sizeof(tampon) - n,
                "INFO tours=%d points=%d vitesse_lidar=%d deg/s erreurs_crc=%d tours_lus=%lu memoire=%d\nFIN\n",
                toursEnregistres, nbMesures, (int)ld06.getSpeed(), (int)ld06.getChecksumFailCount(),
                (unsigned long)(toursLus - toursLusDebut), maxMesures);
  client.write((const uint8_t*)tampon, n);
  Serial.printf("Scan envoye : %d tours, %d points (%lu tours lus pendant le scan, memoire %d points)\n",
                toursEnregistres, nbMesures, (unsigned long)(toursLus - toursLusDebut), maxMesures);
  dernierOrdre = millis();
}

// Broches mises à LOW au démarrage (moteurs à l'arrêt), y compris les 8 pins des moteurs.
// 32 / 33 (PWM et RX du lidar) n'y sont PAS : mettre 33 en sortie avant Serial1.begin() empêchait la réception.
const int BROCHES_LIBRES[] = {0, 2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27};

void setup() {
  // 1. EN TOUT PREMIER : broches à LOW, donc moteurs à l'arrêt (sinon une roue peut tourner seule)
  for (int b : BROCHES_LIBRES) {
    pinMode(b, OUTPUT);
    digitalWrite(b, LOW);
  }

  Serial.begin(115200);
  Serial.println("Moteurs arretes");

  // 2. Mémoire pour les scans (si le tas manque de place, on essaie plus petit)
  for (maxMesures = MAX_MESURES; maxMesures >= 1000; maxMesures -= 1000) {
    mesures = (Mesure*)malloc(maxMesures * sizeof(Mesure));
    if (mesures != nullptr) break;
  }
  if (mesures == nullptr) maxMesures = 0;
  Serial.printf("Memoire pour %d points par scan\n", maxMesures);

  // 3. Lidar : comme l'exemple du prof
  // Serial1 ouvert UNE seule fois, directement sur la broche 33, comme dans test_lidar.ino (qui marche).
  // Pas de ld06.init() : il rouvre Serial1 sans broche, et le lidar n'était plus lu (0 deg/s, 0 erreur CRC).
  Serial1.setRxBufferSize(4096);
  Serial1.begin(LIDAR_BAUD, SERIAL_8N1, RX_LIDAR, TX_LIDAR);
  pinMode(LD06_PWM_PIN, OUTPUT);     // ce que fait ld06.init() pour la broche PWM
  digitalWrite(LD06_PWM_PIN, HIGH);

  // 4. WiFi
  WiFi.softAP(SSID, MDP);
  serveur.begin();
  Serial.print("Reseau WiFi : ");
  Serial.print(SSID);
  Serial.print("   IP du robot : ");
  Serial.println(WiFi.softAPIP());   // 192.168.4.1
}

void loop() {
  bool nouveauTour = lireLidar();    // on lit le lidar en continu, sinon son tampon déborde
  if (toursVoulus > 0) {             // scan en cours : on ne fait que ça
    suivreScan(nouveauTour);
    return;
  }

  if (serveur.hasClient()) {         // PC (re)connecté : la nouvelle connexion remplace l'ancienne
    if (client) client.stop();
    client = serveur.available();
    client.setNoDelay(true);
    Serial.println("PC connecte");
  }

  while (client && client.connected() && client.available()) {
    char c = client.read();
    if (c == 'S') {                  // scan : le chiffre qui suit = nombre de tours
      uint32_t t = millis();
      while (!client.available() && millis() - t < 100) delay(1);
      demarrerScan(constrain(client.read() - '0', 1, 9));
      return;                        // la suite du scan se fait dans les prochains loop()
    } else {
      executer(c);
    }
  }

  if (mouvement != ' ' && millis() - dernierOrdre > TIMEOUT_MS) {
    executer(' ');                   // plus d'ordre : arrêt de sécurité
  }

  static uint32_t dernierAffichage = 0;  // diagnostic dans le moniteur série
  if (millis() - dernierAffichage > 2000) {
    dernierAffichage = millis();
    Serial.printf("Lidar : %d deg/s (3600 = 10 tours/s), %lu tours lus, %d points/tour, %d erreurs CRC | PC %s\n",
                  (int)ld06.getSpeed(), (unsigned long)toursLus, (int)pointsDernierTour,
                  (int)ld06.getChecksumFailCount(), (client && client.connected()) ? "connecte" : "non connecte");
  }
}