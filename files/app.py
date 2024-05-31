from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO
import mysql.connector as con
from typing import Dict
import ffmpeg
from io import BytesIO
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS
import base64
import assemblyai as aai
import auth
import json
import numpy as np
import librosa
import librosa.display


active_users: Dict[str, dict] = {}
current_user = None
user_list = []
chat_id = None


app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("connect")
def handle_connect():
    print(f"User connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("User disconnected")


db = con.connect(host="localhost", user="root", password="", database="ChatAppUserData")


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


def load_audio(file_path):
    # Charger le fichier audio
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


def extract_mfcc(audio, sr, n_mfcc=13):
    # Extraire les coefficients MFCC de l'audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs, mfccs_mean


def calculate_distance(mfcc1_mean, mfcc2_mean):
    # Calculer la distance euclidienne entre deux vecteurs MFCC moyens
    distance = np.linalg.norm(mfcc1_mean - mfcc2_mean)
    return distance


def blob_to_audio(blob):
    # Crée un flux de mémoire pour stocker le blob
    # blob_stream = blob.getvalue()

    # Convertit le flux de mémoire en un format audio WAV
    audio_stream, _ = (
        ffmpeg.input("pipe:")
        .output("pipe:", format="wav")
        .run(input=blob, capture_stdout=True, capture_stderr=True)
    )
    return audio_stream


def text_to_audio_blob(text, lang="fr"):
    # Convertir le texte en audio en utilisant gTTS
    tts = gTTS(text=text, lang=lang)

    # Sauvegarder l'audio dans un buffer en mémoire (Blob)
    audio_blob = BytesIO()
    tts.write_to_fp(audio_blob)

    # Repositionner le pointeur au début du blob
    audio_blob.seek(0)
    print(f"dans la fonction de synthese vocale {type(audio_blob.getvalue())}")
    #audio_base64 = base64.b64encode(audio_blob.getvalue()).decode('utf-8')
    audio_b = audio_blob.getvalue()

    return audio_b


def trasai(audio) :
    aai.settings.api_key = "006281e165754264912921b7fe4525c7"
    transcriber = aai.Transcriber()

    #transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/news.mp4")
    transcript = transcriber.transcribe(audio)
    return transcript.text


def transcript(blob):
    audio_wav = blob_to_audio(blob)
    r = sr.Recognizer()

    # Convertit l'audio WAV en un objet AudioData
    audio_data = sr.AudioData(audio_wav, sample_rate=44100, sample_width=2)

    # Utilise l'objet AudioData comme source audio pour la transcription
    try:
        text = r.recognize_google(audio_data, language="fr-FR")
        print("Converting audio transcripts into text ...")
        return text
    
    # except sr.UnknownValueError:
    #     print("Google Speech Recognition could not understand audio")

    # except sr.RequestError as e:
    #     print(
    #         "Could not request results from Google Speech Recognition service; {0}".format(
    #             e
    #         )
        # )

    except Exception as e:
        print("Sorry.. run again due to error: ", str(e))

@app.route("/chat")
def chat():
    return render_template("index.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    global current_user
    error = None
    if request.method == "POST":
        userinfos = request.form
        username = userinfos["username"]
        password = userinfos["password"]
        cursor = db.cursor()
        cursor.execute("SELECT * FROM User WHERE Name = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            if user[2] == password:
                current_user = username
                session["username"] = username
                return redirect("chat")
            else:
                error = "Invalid password. Please try again."
                print(error)
        else:
            error = "Username does not exist. Please try again."
            print(error)
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        userinfos = request.form
        username = dict(userinfos)
        username = userinfos["username"]
        password = userinfos["password"]
        cursor = db.cursor()
        cursor.execute("SELECT * FROM User WHERE Name = %s", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            error = "Username already exists. Please choose a different username."
            print(error)
        else:
            try:
                cursor.execute(
                    "INSERT INTO User (Name, Password) VALUES (%s, %s)",
                    (username, password),
                )
                db.commit()
                return redirect(url_for("login"))
            except Exception as e:
                db.rollback()
                error = "Error occurred during registration. Please try again."
                print(e)
            finally:
                cursor.close()

    return render_template("register.html", error=error)


@app.route("/fetch_user", methods=["GET", "POST"])
def fetch_users():
    global user_list
    file = open("users.txt", "r")
    data = file.readlines()
    user_list = data
    return redirect("chat")


# @app.route("update_cid/<string: chat_id>", methods=["GET", "POST"])
# def update_cid():
#     global chat_id
#     chat_id


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    global current_user, user_list, chat_id
    return render_template(
        "dashboard.html", username=current_user, user_list=user_list, chat_id=chat_id
    )


@app.route("/auth_voc", methods=["GET", "POST"])
def voc_auth():
    if request.method == "POST":
        voice_name = request.form["voiceName"]
        mfcc_mean = session.get("sus_mfcc_mean")
        if mfcc_mean:
            # Insertion du nouveau nom et des MFCC dans la base de données
            cursor = db.cursor()
            insert_query = "INSERT INTO Voc_auth (nom, mfcc) VALUES (%s, %s)"
            cursor.execute(insert_query, (voice_name, json.dumps(mfcc_mean.tolist())))
            db.commit()
            # Redirection vers le chat après l'ajout
            return redirect("/chat")
        else:
            return "MFCC data not found in session", 400
    else:
        return render_template("authent.html")


@socketio.on("message")
def handle_message(data):
    if "message" in data:
        socket_id = data["socket_id"]
        username = data["username"]
        message = f"message de {username}. {data['message']}"
        blob = text_to_audio_blob(message)
        print(f"Message from {username} [{socket_id}] : {message} ")
        socketio.emit(
            "message",
            {"message": data['message'], "socket_id": socket_id, "username": username, "audio_base64": blob},
        )

    elif "audio" in data:
        audiob = data["audio"]
        audio = blob_to_audio(audiob)

        sus_audio, sus_sr = load_audio(audio)
        sus_mfss, sus_mfcc_mean = extract_mfcc(sus_audio, sus_sr)
    
        # Requête pour récupérer le tableau
        cursor = db.cursor()
        cursor.execute("SELECT nom, mfcc FROM Voc_auth")
        results = cursor.fetchall()
    
        min_distance = float('inf')
        min_nom = None

        if not results:  # Vérifier si la base de données est vide
            print("Aucune correspondance trouvée.")
            session["sus_mfcc_mean"] = sus_mfcc_mean.tolist()
            socketio.emit('redirect', {'url': '/auth_voc'})
        else:
            for row in results:
                nom = row[0]
                mfcc = row[1]
                mfcc_array = np.array(json.loads(mfcc))
                distance = np.linalg.norm(sus_mfcc_mean - mfcc_array)

                if distance < min_distance and distance < 50:
                    min_distance = distance
                    min_nom = nom

            if min_nom is None:
                print("Aucune correspondance trouvée.")
                session["sus_mfcc_mean"] = sus_mfcc_mean.tolist()
                socketio.emit('redirect', {'url': '/auth_voc'})
            else:
                print(f"Nom avec la distance minimale : {min_nom} (distance : {min_distance})")



                socket_id = data["socket_id"]
                username = min_nom
                text = transcript(audiob)
                print(f"Audio from {username} [{socket_id}] : {len(audiob)} bytes")
                print(f"Text transcrit : {text} ")

                socketio.emit(
                    "message",
                    {
                        "text": text,
                        "audio": audiob,
                        "socket_id": socket_id,
                        "username": username,
                    },
                )



if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
