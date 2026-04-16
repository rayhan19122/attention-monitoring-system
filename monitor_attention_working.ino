// ============================================================
// ESP32-CAM Smart Classroom Attention Monitor
// ============================================================
// HOW IT WORKS:
//  1. Connects to your WiFi
//  2. Initializes the OV2640 camera
//  3. Every 10 seconds: takes a photo and sends it to your
//     Flask server via HTTP POST
// ============================================================

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ─── CONFIGURE THESE ────────────────────────────────────
const char* ssid     = "Rayhan";      // Your WiFi name
const char* password = "244466666";  // Your WiFi password
const char* serverUrl = "http://192.168.0.100:5001/upload";
// ↑ Replace 192.168.1.5 with your laptop's IP address
// ─────────────────────────────────────────────────────────

// AI-Thinker ESP32-CAM pin definitions
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;  // 20 MHz clock
  config.pixel_format = PIXFORMAT_JPEG;  // JPEG = smaller file
  config.frame_size   = FRAMESIZE_VGA;   // 640x480 pixels
  config.jpeg_quality = 12;  // 0=best, 63=worst. 12 is good balance
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init FAILED: 0x%x\n", err);
    return;
  }
  Serial.println("Camera initialized OK");
}

void sendPhoto() {
  // Step 1: Capture a frame from the camera
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed!");
    return;
  }
  Serial.printf("Photo captured: %d bytes\n", fb->len);

  // Step 2: Connect to Flask server and send the image
  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "image/jpeg");

  // fb->buf = raw JPEG bytes, fb->len = number of bytes
  int httpCode = http.POST(fb->buf, fb->len);

  if (httpCode == 200) {
    Serial.println("Image sent successfully!");
    Serial.println(http.getString());  // Print server response
  } else {
    Serial.printf("HTTP Error: %d\n", httpCode);
  }

  http.end();
  esp_camera_fb_return(fb);  // Free the frame buffer memory
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n\n=== Smart Classroom Monitor Starting ===");

  // Initialize camera
  initCamera();

  // Connect to WiFi
  Serial.printf("Connecting to WiFi: %s\n", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  delay(2000);
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n--- Taking photo ---");
    sendPhoto();
  } else {
    Serial.println("WiFi lost, reconnecting...");
    WiFi.reconnect();
  }
  delay(10000);  // Wait 10 seconds before next capture
}