from flask import Flask, request, jsonify
import whisper
import torch
import logging
import tempfile
import os
import gc
import subprocess

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
FORCE_CPU  = os.getenv("FORCE_CPU", "0") == "1"
device     = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device} / model: {MODEL_NAME}")

model = whisper.load_model(MODEL_NAME, device=device)

@app.route("/ping", methods=["GET"])
def ping():
    try:
        return jsonify({"ok": True}), 200
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as temp_in:
            temp_input_path = temp_in.name
            audio_file.save(temp_input_path)

        temp_output_path = tempfile.mktemp(suffix=".wav")

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        result = model.transcribe(temp_output_path, language="ru")

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return jsonify({"text": result["text"]})

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e}")
        return jsonify({"error": "Audio conversion failed"}), 500

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4000)
