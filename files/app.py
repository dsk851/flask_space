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
import json
import numpy as np
import librosa
import librosa.display


active_users: Dict[str, dict] = {}
current_user = None
chat_id = None


app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("connect")
def handle_connect():
    print("User connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("User disconnected")


db = con.connect(host="localhost", user="root", password="", database="ChatAppUserData")


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


def blob_to_audio(blob):
    audio_stream, _ = (
        ffmpeg.input("pipe:")
        .output("pipe:", format="wav")
        .run(input=blob, capture_stdout=True, capture_stderr=True)
    )
    return audio_stream


def text_to_audio_blob(text, lang="fr"):
    tts = gTTS(text=text, lang=lang)

    audio_blob = BytesIO()
    tts.write_to_fp(audio_blob)

    audio_blob.seek(0)
    audio_b = audio_blob.getvalue()

    return audio_b


def trasai(audio):
    aai.settings.api_key = "006281e165754264912921b7fe4525c7"
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(audio)
    return transcript.text


def transcript(blob):
    audio_wav = blob_to_audio(blob)
    r = sr.Recognizer()

    audio_data = sr.AudioData(audio_wav, sample_rate=44100, sample_width=2)

    try:
        text = r.recognize_google(audio_data, language="fr-FR")
        print("Converting audio transcripts into text ...")
        return text

    except Exception as e:
        print("Sorry.. run again due to error: ", str(e))


@app.route("/chat")
def chat():
    username = session.get('username', 'Guest') 
    return render_template("index.html", username = username)


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
            {
                "message": data["message"],
                "socket_id": socket_id,
                "username": username,
                "audio_base64": blob,
            },
        )

    elif "audio" in data:
        audiob = data["audio"]
        socket_id = data["socket_id"]
        username = data["username"]
        text = transcript(audiob)
        print(f"Audio from {username} [{socket_id}] : {len(audiob)} bytes")
        print(f"Text transcrit : {text}")
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
