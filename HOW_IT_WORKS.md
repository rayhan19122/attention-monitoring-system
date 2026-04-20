# Smart Classroom Attention Monitor — How It Works

## What Does This Project Do?

This project uses a **camera** and **AI** to check if students in a classroom are paying attention or not. It counts how many students are **attentive** vs **distracted** and shows the results on a **live dashboard**.

---

## The Big Picture (3 Simple Steps)

```
📷 ESP32-CAM        →      🖥️ Flask Server      →      ☁️ Firebase + Dashboard
(Takes photos)             (AI analyzes faces)          (Shows results live)
```

1. A small camera (ESP32-CAM) takes a photo of the classroom **every 10 seconds**
2. The photo is sent to a Python server, which uses **AI to detect faces** and figure out who is paying attention
3. The results are saved to **Firebase** (cloud database) and shown on a **web dashboard**

---

## The 4 Parts Explained

### 1. ESP32-CAM (The Camera) — `monitor_attention_working.ino`

This is a small microcontroller with a built-in camera. It does 3 things:

- Connects to your **WiFi**
- Takes a **JPEG photo** (640×480 pixels) every 10 seconds
- Sends the photo to the Flask server using **HTTP POST**

Think of it like a security camera that emails you a photo every 10 seconds.

---

### 2. Flask Server (The Brain) — `app.py`

This is a Python web server running on your laptop. When it receives a photo:

1. **Saves** the image to the `uploads/` folder
2. Passes it to the **AI module** for analysis
3. **Stores the result** in Firebase (cloud database)
4. Sends back a "success" message to the ESP32-CAM

The server runs on `http://your-laptop-ip:5001`.

---

### 3. AI Attention Detector — `attention_detector.py`

This is the core AI part. It uses **MediaPipe FaceMesh** (by Google) to detect faces and analyze them. Here's how it decides if someone is attentive:

#### Step A: Find Faces
- MediaPipe scans the image and finds up to **10 faces**
- For each face, it maps **468 landmark points** (eyes, nose, mouth, etc.)

#### Step B: Check Eyes (Eye Aspect Ratio — EAR)
- It measures how **open** or **closed** the eyes are
- **EAR > 0.20** → Eyes are open ✅
- **EAR < 0.20** → Eyes are closed (sleeping/looking down) ❌

```
EAR = (vertical eye distance) / (horizontal eye distance)
```

#### Step C: Check Head Direction
- It checks if the person's **head is facing forward** by looking at the nose position relative to the eyes
- Nose roughly centered between the eyes → Facing forward ✅
- Nose way off to the side → Looking away ❌
- It also checks if the head is **nodding down** too much

#### Final Decision
A student is **attentive** only if **BOTH** conditions are true:
- Eyes are open ✅ **AND**
- Head is facing forward ✅

Otherwise → **Distracted**

#### Output Example
```json
{
  "attentive": 3,
  "distracted": 1,
  "total_faces": 4,
  "score": 75.0
}
```
This means: 4 students detected, 3 paying attention, 1 distracted → 75% attention score.

---

### 4. Dashboard (The Display) — `dashboard.html`

A web page that connects directly to Firebase and shows:

- **Live attention score** (big circle with percentage)
- **Attentive vs Distracted count**
- **Trend chart** (attention over time)
- **Distribution chart** (pie chart)
- **Recent readings table**
- **Time filters** (Live, Last Hour, Last Day, Last Week, Custom)
- **Alert banner** when attention drops below 40%

Just open `dashboard.html` in a browser — no server needed for this file. It reads data directly from Firebase.

---

## How Everything Connects

```
┌──────────────┐     photo (JPEG)     ┌──────────────────┐
│  ESP32-CAM   │ ──────────────────►  │   Flask Server   │
│  (Arduino)   │     HTTP POST        │   (app.py)       │
└──────────────┘                      └────────┬─────────┘
                                               │
                                        calls AI module
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │ attention_detector│
                                      │    (MediaPipe)   │
                                      └────────┬─────────┘
                                               │
                                         returns result
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │     Firebase     │
                                      │  (Cloud Database)│
                                      └────────┬─────────┘
                                               │
                                          live sync
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │    Dashboard     │
                                      │  (dashboard.html)│
                                      └──────────────────┘
```

---

## How to Run It

### Requirements
- Python 3.x with packages: `flask`, `opencv-python`, `mediapipe`, `firebase-admin`
- ESP32-CAM board (AI-Thinker model)
- Firebase project with Realtime Database
- `serviceAccountKey.json` from Firebase Console

### Steps
1. **Start the server:**
   ```bash
   python app.py
   ```
2. **Upload the Arduino code** to ESP32-CAM (update WiFi name, password, and server IP)
3. **Open `dashboard.html`** in your browser to see the live results

---

## File Summary

| File | What It Does |
|------|-------------|
| `app.py` | Flask web server — receives photos, runs AI, saves to Firebase |
| `attention_detector.py` | AI module — detects faces and checks attention using eye + head analysis |
| `firebase_config.py` | Helper to initialize Firebase connection |
| `dashboard.html` | Live web dashboard — shows charts and stats from Firebase |
| `monitor_attention_working.ino` | ESP32-CAM Arduino code — captures and sends photos |
| `serviceAccountKey.json` | Firebase authentication credentials |
| `requirements.txt` | Python package list |
| `uploads/` | Folder where received photos are saved |
