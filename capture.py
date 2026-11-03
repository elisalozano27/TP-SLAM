# capture.py - Enregistre plusieurs tours du lidar LD06 dans un fichier CSV
# Fonctionne avec le code Arduino du prof (printScanTeleplot + printScanCSV)

import serial  # pour lire le port série (USB)

PORT = "COM3"           # port de l'ESP32 (voir Outils > Port dans l'IDE Arduino)
FICHIER = "scan_01.csv" # à changer pour chaque nouveau scan
NB_TOURS = 20            # nombre de tours de lidar à enregistrer

# Ouverture du port à la même vitesse que le code Arduino (115200)
ser = serial.Serial(PORT, 115200, timeout=2)
ser.reset_input_buffer()  # on vide les anciennes données pour partir d'un scan complet

with open(FICHIER, "w") as f:
    f.write("tour,angle_deg,distance_mm,intensite\n")

    for tour in range(1, NB_TOURS + 1):
        # On attend la ligne Teleplot ">lidar:..." : les points du tour arrivent juste après
        while not ser.readline().decode(errors="ignore").startswith(">lidar:"):
            pass

        # On lit les points jusqu'à la ligne vide (fin du tour)
        while True:
            ligne = ser.readline().decode(errors="ignore").strip()
            if ligne == "":
                break
            valeurs = ligne.split(",")   # N, Angle, Distance, Intensity, x, y
            if len(valeurs) == 6 and valeurs[0].isdigit():   # on ignore l'en-tête et les lignes abîmées
                f.write(f"{tour},{valeurs[1]},{valeurs[2]},{valeurs[3]}\n")

        print("Tour", tour, "/", NB_TOURS, "enregistré")

ser.close()  # on libère le port
print("Scan enregistré dans", FICHIER)