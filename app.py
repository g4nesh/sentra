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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Sapling AI API configuration
SAPLING_API_KEY = "2L93SILP3SE7YJ2934YN1JN8CIH41K91"
SAPLING_API_URL = "https://api.sapling.ai/api/v1/aidetect"

def check_ai_content(text):
    """Check if the text is AI-generated using Sapling AI API."""
    try:
        headers = {
            "Content-Type": "application/json",
            "apikey": SAPLING_API_KEY
        }
        data = {
            "text": text
        }
        response = requests.post(SAPLING_API_URL, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result
    except Exception as e:
        logger.error(f"Error checking AI content: {e}")
        return None

# Try to load the model
try:
    logger.info("Loading Whisper model...")
    # Use faster-whisper with a smaller model for speed
    model = WhisperModel(
        "tiny",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="int8"  # Changed from float16 to int8 for better compatibility
    )
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    raise

ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'm4a', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_features(audio_path):
    """Extract audio features for speaker identification."""
    waveform, sample_rate = torchaudio.load(audio_path)
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sample_rate

@app.route('/')
def index():
    return render_template('index.html')

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
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing file: {filename}")
        
        # Transcribe with speaker diarization
        segments, info = model.transcribe(
            filepath,
            language='en',
            task='transcribe',
            beam_size=5,
            vad_filter=True,  # Voice Activity Detection
            vad_parameters=dict(min_silence_duration_ms=500),  # Adjust silence threshold
        )
        
        # Process segments and organize by speaker
        speakers = {}
        
        for segment in segments:
            # Get speaker ID (0 or 1)
            speaker_id = segment.speaker if hasattr(segment, 'speaker') else 0
            
            # Initialize speaker if not exists
            if speaker_id not in speakers:
                speakers[speaker_id] = []
            
            text = segment.text.strip()
            if text:
                # Check for AI-generated content
                ai_check = check_ai_content(text)
                ai_score = ai_check.get('score', 0) if ai_check else 0
                
                # Add timestamp, text, and AI detection results
                speakers[speaker_id].append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': text,
                    'ai_score': ai_score,
                    'is_ai_generated': ai_score > 0.5 if ai_check else False
                })
        
        # Clean up the temporary file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'speakers': speakers
        })
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    try:
        logger.info("Starting server...")
        app.run(debug=True, host='0.0.0.0', port=4000)
    except Exception as e:
        logger.error(f"Server error: {e}")

