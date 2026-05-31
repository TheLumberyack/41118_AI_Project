#include <Servo.h>

Servo myServo;
const int SERVO_PIN = 10;

void setup() {
  Serial.begin(9600);
  myServo.attach(SERVO_PIN);
  myServo.write(0);
  delay(500);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'D') {
      for (int i = 0; i < 5; i++) {
        myServo.write(90);
        delay(500);
        myServo.write(0);
        delay(500);
      }
    }
  }
}