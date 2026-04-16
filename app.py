# ============================================================
# Flask Backend Server — Smart Classroom Monitor
# Run with: python app.py
# ============================================================

from flask import Flask, request, jsonify
import os
import time
import firebase_admin
from firebase_admin import credentials, db
from attention_detector import analyze_image  # Our AI module

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Allow up to 10MB

# ─── FIREBASE SETUP ──────────────────────────────────────────
# Replace with your Firebase project details
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://attention-detector-default-rtdb.firebaseio.com'
})
# ─────────────────────────────────────────────────────────────

# Create folder to store received images
os.makedirs("uploads", exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    This endpoint is called by the ESP32-CAM.
    It receives a JPEG image, processes it with AI,
    and stores the result in Firebase.
    """
    # Step 1: Get the raw image data from the request
    image_data = request.data
    if not image_data:
        return jsonify({"error": "No image data received"}), 400
    
    # Add minimum size guard — reject suspiciously small or corrupt images
    if len(image_data) < 5000:
        return jsonify({"error": "Image too small, likely corrupt"}), 400

    # Step 2: Save image to disk with a timestamp filename
    timestamp = int(time.time())
    filename = f"uploads/image_{timestamp}.jpg"
    with open(filename, 'wb') as f:
        f.write(image_data)
    print(f"[{timestamp}] Image saved: {filename}")
    
    # Wrap AI call in try/except so the server never crashes
    try:
        result = analyze_image(filename)
    except Exception as e:
        print(f"[ERROR] analyze_image failed: {e}")
        return jsonify({"error": str(e)}), 500

    # Step 3: Run AI detection on the saved image
    result = analyze_image(filename)
    print(f"AI Result: {result}")

    # Step 4: Save result to Firebase Realtime Database
    ref = db.reference(f"classroom/session_001/{timestamp}")
    ref.set({
        "timestamp": timestamp,
        "attentive": result["attentive"],
        "distracted": result["distracted"],
        "attention_score": result["score"],
        "total_faces": result["total_faces"]
    })

    # Step 5: Send success response back to ESP32
    return jsonify({
        "status": "success",
        "result": result
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Simple check to confirm server is running"""
    return jsonify({"status": "running"}), 200

if __name__ == '__main__':
    print("=== Smart Classroom Server Starting ===")
    print("Waiting for images from ESP32-CAM...")
    app.run(host='0.0.0.0', port=5001, debug=True)