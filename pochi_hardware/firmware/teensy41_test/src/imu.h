#pragma once

#include <Arduino.h>
#include <ICM_20948.h>
#include <SPI.h>

#include <cmath>
#include <cstdint>

constexpr uint8_t IMU_SPI_MOSI_PIN = 26;
constexpr uint8_t IMU_SPI_MISO_PIN = 39;
constexpr uint8_t IMU_SPI_SCK_PIN = 27;
constexpr uint8_t IMU_SPI_CS_PIN = 38;

enum : uint32_t {
  IMU_STATE_INITIALIZED = 1U << 0,
  IMU_STATE_SAMPLE_VALID = 1U << 1,
  IMU_STATE_QUAT6 = 1U << 2,
};

struct ImuTelemetry {
  uint32_t flags = IMU_STATE_QUAT6;
  uint32_t last_rx_age_us = UINT32_MAX;
  float quaternion_w = NAN;
  float quaternion_x = NAN;
  float quaternion_y = NAN;
  float quaternion_z = NAN;
  float acceleration_x = NAN;
  float acceleration_y = NAN;
  float acceleration_z = NAN;
  float angular_velocity_x = NAN;
  float angular_velocity_y = NAN;
  float angular_velocity_z = NAN;
  float temperature_c = NAN;
  uint32_t sample_counter = 0;
  uint32_t accuracy = 0;
  uint32_t reserved = 0;
};

namespace imu_detail {

inline ICM_20948_SPI device;
inline ImuTelemetry telemetry;
inline uint32_t last_sample_us = 0;

inline bool configure_dmp() {
  bool success = true;
  success &= device.initializeDMP() == ICM_20948_Stat_Ok;
  success &= device.enableDMPSensor(
                 INV_ICM20948_SENSOR_GAME_ROTATION_VECTOR) == ICM_20948_Stat_Ok;
  success &= device.setDMPODRrate(DMP_ODR_Reg_Quat6, 0) == ICM_20948_Stat_Ok;
  success &= device.enableFIFO() == ICM_20948_Stat_Ok;
  success &= device.enableDMP() == ICM_20948_Stat_Ok;
  success &= device.resetDMP() == ICM_20948_Stat_Ok;
  success &= device.resetFIFO() == ICM_20948_Stat_Ok;
  return success;
}

}  // namespace imu_detail

inline bool imu_init() {
  using namespace imu_detail;
  telemetry = {};
  telemetry.flags = IMU_STATE_QUAT6;

  SPI1.begin();
  SPI1.setMOSI(IMU_SPI_MOSI_PIN);
  SPI1.setMISO(IMU_SPI_MISO_PIN);
  SPI1.setSCK(IMU_SPI_SCK_PIN);

  bool detected = false;
  for (uint8_t attempt = 0; attempt < 3 && !detected; ++attempt) {
    device.begin(IMU_SPI_CS_PIN, SPI1);
    detected = device.status == ICM_20948_Stat_Ok;
    if (!detected) {
      delay(50);
    }
  }
  if (!detected || !configure_dmp()) {
    return false;
  }

  telemetry.flags |= IMU_STATE_INITIALIZED;
  return true;
}

inline void imu_loop() {
  using namespace imu_detail;
  if ((telemetry.flags & IMU_STATE_INITIALIZED) == 0U) {
    return;
  }

  icm_20948_DMP_data_t data{};
  device.readDMPdataFromFIFO(&data);
  if (device.status != ICM_20948_Stat_Ok &&
      device.status != ICM_20948_Stat_FIFOMoreDataAvail) {
    return;
  }
  if ((data.header & DMP_header_bitmap_Quat6) == 0U) {
    return;
  }

  constexpr double kQuatScale = 1073741824.0;
  const double x = static_cast<double>(data.Quat6.Data.Q1) / kQuatScale;
  const double y = static_cast<double>(data.Quat6.Data.Q2) / kQuatScale;
  const double z = static_cast<double>(data.Quat6.Data.Q3) / kQuatScale;
  const double w_squared = 1.0 - (x * x + y * y + z * z);

  telemetry.quaternion_w = static_cast<float>(std::sqrt(w_squared > 0.0 ? w_squared : 0.0));
  telemetry.quaternion_x = static_cast<float>(x);
  telemetry.quaternion_y = static_cast<float>(y);
  telemetry.quaternion_z = static_cast<float>(z);
  telemetry.sample_counter++;
  telemetry.flags |= IMU_STATE_SAMPLE_VALID;
  last_sample_us = micros();
}

inline ImuTelemetry imu_get_telemetry() {
  auto result = imu_detail::telemetry;
  result.last_rx_age_us =
      (result.flags & IMU_STATE_SAMPLE_VALID) != 0U
          ? static_cast<uint32_t>(micros() - imu_detail::last_sample_us)
          : UINT32_MAX;
  return result;
}
