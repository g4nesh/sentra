from flask import Flask, render_template, request, jsonify
import whisper
import logging
import os
from werkzeug.utils import secure_filename
import tempfile
import torch
import numpy as np
from faster_whisper import WhisperModel
import torchaudio
import requests

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

SAPLING_API_KEY = "2L93SILP3SE7YJ2934YN1JN8CIH41K91"
SAPLING_API_URL = "https://api.sapling.ai/api/v1/aidetect"

def check_ai_content(text):
    try:
        headers = {
            "Content-Type": "application/json",
            "apikey": SAPLING_API_KEY
        }
        data = { "text": text }
        response = requests.post(SAPLING_API_URL, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error checking AI content: {e}")
        return None

try:
    logger.info("Loading Whisper model...")
    model = WhisperModel(
        "tiny",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="int8"
    )
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    raise

ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'm4a', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mic-activity')
def mic_activity():
    return render_template('mic_activity.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        logger.info(f"Processing file: {filename}")

        segments, info = model.transcribe(
            filepath,
            language='en',
            task='transcribe',
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        speakers = {}
        for segment in segments:
            speaker_id = segment.speaker if hasattr(segment, 'speaker') else 0
            if speaker_id not in speakers:
                speakers[speaker_id] = []

            text = segment.text.strip()
            if text:
                ai_check = check_ai_content(text)
                ai_score = ai_check.get('score', 0) if ai_check else 0

                speakers[speaker_id].append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': text,
                    'ai_score': ai_score,
                    'is_ai_generated': ai_score > 0.5 if ai_check else False
                })

        os.remove(filepath)
        return jsonify({ 'success': True, 'speakers': speakers })

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    try:
        logger.info("Starting server...")
        app.run(debug=True, host='0.0.0.0', port=4000)
    except Exception as e:
        logger.error(f"Server error: {e}")
