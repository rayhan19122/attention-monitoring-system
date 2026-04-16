import firebase_admin
from firebase_admin import credentials, db

def init_firebase():
    # Load service account key downloaded from Firebase console
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://attention-detector-default-rtdb.firebaseio.com'
    })
    print("Firebase initialized!")

def store_result(result):
    ref = db.reference('/classroom_data')
    # Store in history (timestamped log)
    ref.child('history').push(result)
    # Update current reading for live dashboard
    ref.child('current').set(result)