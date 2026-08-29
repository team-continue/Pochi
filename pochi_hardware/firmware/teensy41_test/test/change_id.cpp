#include <Arduino.h>
#include <FlexCAN_T4.h>

#include "robostride.h"

#define OLD_ID 1
#define NEW_ID 2

FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16> can;
RoboStride<CAN3, RX_SIZE_256, TX_SIZE_16> motor(
    &can, 0xFD, OLD_ID, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f);

void setup() {
  Serial.begin(115200);
  can.begin();
  can.setBaudRate(1000000);
  delay(500);
  motor.Set_CAN_ID(NEW_ID);
  Serial.printf("RoboStride ID: %u -> %u\n", OLD_ID, NEW_ID);
}

void loop() {}
