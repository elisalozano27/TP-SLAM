// robot_slam_esp32.ino - Jour 3 : pilotage en WiFi (comme pilotage_wifi.ino) + scans du lidar à la demande
// L'ESP32 crée son propre réseau WiFi (SSID / MDP) et attend le PC (robot_slam.py) sur le port PORT.
// Le PC envoie une lettre par ordre :
//   z avance, s recule, q tourne à gauche, d tourne à droite, a translation gauche, e translation droite,
//   espace = arrêt, p / m = plus vite / moins vite,
//   S suivi d'un chiffre (ex. "S2") = le robot s'arrête et envoie 2 tours du lidar.
// Réponse à un scan : une ligne par point "tour,angle_deg,distance_mm,intensite", puis une ligne "FIN".
// Sécurité : sans nouvel ordre pendant TIMEOUT_MS (touche relâchée, WiFi coupé...), le robot s'arrête.
// Moteurs pilotés directement en PWM par l'ESP32 : 2 pins par roue (pas de bibliothèque MecanumCar).
// Bibliothèque LD06 du prof installée (LD06.zip), câblage lidar comme les jours 1 et 2.

#include <WiFi.h>
#include "ld06.h"

const char* SSID = "Robot_SLAM";     // nom du réseau créé par le robot
const char* MDP = "eigsi2026";       // mot de passe (8 caractères minimum)
const int PORT = 3333;
const uint32_t TIMEOUT_MS = 300;     // arrêt automatique sans ordre pendant ce temps
const uint32_t ATTENTE_SCAN_MS = 300;  // avant de mesurer, le robot reste arrêté ce temps (fin des secousses)

// Lidar : mêmes réglages que l'exemple du prof
#define RX_LIDAR 33
#define TX_LIDAR -1
#define LD06_PWM_PIN 32
#define LIDAR_BAUD 230400
LD06 ld06(Serial1, LD06_PWM_PIN);

// Roues : {pin qui fait AVANCER le robot, pin qui le fait RECULER}, relevées avec test_pins.ino.
// Si une roue tourne à l'envers quand on appuie sur z : échanger ses deux pins ici.
const int ROUES[4][2] = {
  {19, 21},    // avant gauche   (19 trigo, 21 horaire)
  {18, 5},     // avant droite   (18 horaire, 5 trigo)
  {2, 4},      // arrière gauche (4 trigo, 2 horaire)
  {17, 16},    // arrière droite (17 horaire, 16 trigo)
};
WiFiServer serveur(PORT);
WiFiClient client;

char mouvement = ' ';                // mouvement en cours (' ' = arrêté)
uint32_t dernierOrdre = 0;           // instant du dernier ordre reçu (ms)
int vitesse = 150;                   // rapport cyclique PWM des roues (50 à 255)

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
    default: return;                   // caractère inconnu : ignoré
  }
  mouvement = c;
  dernierOrdre = millis();
}

void scanner(int nbTours) {
  executer(' ');                                   // on s'arrête
  uint32_t t0 = millis();
  while (millis() - t0 < ATTENTE_SCAN_MS) {        // on attend sans bouger, en continuant à lire le lidar
    ld06.readScan();
  }
  // Le tour 0 a pu commencer avant l'arrêt complet : on le jette, puis on envoie les nbTours suivants
  char ligne[40];
  int tour = 0;
  t0 = millis();
  while (tour <= nbTours && millis() - t0 < 3000) {   // 3 s maximum (lidar débranché...)
    if (!ld06.readScan()) {
      delay(1);
      continue;
    }
    if (tour > 0) {
      String bloc;                                 // tout le tour d'un coup : plus rapide en WiFi
      bloc.reserve(12000);
      for (uint16_t i = 0; i < ld06.getNbPointsInScan(); i++) {
        DataPoint* p = ld06.getPoints(i);
        snprintf(ligne, sizeof(ligne), "%d,%.2f,%u,%u\n", tour, p->angle, p->distance, p->intensity);
        bloc += ligne;
      }
      client.print(bloc);
    }
    tour++;
  }
  client.print("FIN\n");
}

// Broches mises à LOW au démarrage (comme ton code de reset), y compris les 8 pins des moteurs.
// 32 / 33 (PWM et RX du lidar) sont reconfigurées juste après par Serial1.begin() et ld06.init().
const int BROCHES_LIBRES[] = {0, 2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33};

void setup() {
  // 1. EN TOUT PREMIER : broches à LOW, donc moteurs à l'arrêt (sinon une roue peut tourner seule)
  for (int b : BROCHES_LIBRES) {
    pinMode(b, OUTPUT);
    digitalWrite(b, LOW);
  }
  // (plus de mecanumCar.Stop() ni left_led / right_led : les moteurs ne sont plus en I2C,
  //  et ces fonctions feraient clignoter 19 / 21, donc tourner la roue avant gauche)

  // 2. Le reste
  Serial.begin(115200);
  Serial.println("Moteurs arretes");
  Serial1.setRxBufferSize(4096);
  Serial1.begin(LIDAR_BAUD, SERIAL_8N1, RX_LIDAR, TX_LIDAR);
  Serial1.setTimeout(10);
  ld06.init();
  WiFi.softAP(SSID, MDP);
  serveur.begin();
  Serial.print("Reseau WiFi : ");
  Serial.print(SSID);
  Serial.print("   IP du robot : ");
  Serial.println(WiFi.softAPIP());   // 192.168.4.1
}

void loop() {
  ld06.readScan();                   // on lit le lidar en continu, sinon son tampon déborde
  if (!client || !client.connected()) {
    client = serveur.available();    // nouveau PC connecté ?
    if (client) client.setNoDelay(true);
  }
  while (client && client.available()) {
    char c = client.read();
    if (c == 'S') {                  // scan : le chiffre qui suit = nombre de tours
      uint32_t t = millis();
      while (!client.available() && millis() - t < 100) delay(1);
      scanner(constrain(client.read() - '0', 1, 9));
    } else {
      executer(c);
    }
  }
  if (mouvement != ' ' && millis() - dernierOrdre > TIMEOUT_MS) {
    executer(' ');                   // plus d'ordre : arrêt de sécurité
  }
}
