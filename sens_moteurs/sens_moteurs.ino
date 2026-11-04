// test_pins.ino - met chaque pin à 1 pendant 2 s : noter quelle roue tourne, et dans quel sens
const int PINS[] = {2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27};

void setup() {
  Serial.begin(115200);
  for (int p : PINS) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }
  delay(2000);
}

void loop() {
  for (int p : PINS) {
    Serial.print("Pin "); Serial.println(p);
    digitalWrite(p, HIGH);
    delay(2000);
    digitalWrite(p, LOW);
    delay(1000);
  }
  Serial.println("---- fin du tour ----");
}