import gradio as gr
import torch
import os
import subprocess
import sys
import tempfile
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import warnings
import ollama
import time
from datasets import Audio
import numpy as np
import librosa
import re

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated")

# Set the direct path to ffmpeg.exe
current_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(current_dir, "ffmpeg.exe")

# Verify ffmpeg.exe exists and add its directory to PATH
if os.path.exists(ffmpeg_path):
    print(f"Found ffmpeg at: {ffmpeg_path}")
    # Add the directory containing ffmpeg.exe to the PATH
    os.environ["PATH"] = current_dir + os.pathsep + os.environ["PATH"]
else:
    print(f"Warning: ffmpeg.exe not found at: {ffmpeg_path}")
    print("Audio processing may not work correctly.")

# Initialize device settings
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-base"

print("Loading Whisper model...")
# Load model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Don't set forced_decoder_ids here, will use them in generate_kwargs instead
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Create pipeline with better chunk support
whisper = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True,  # Enable timestamps for better chunking
    torch_dtype=torch_dtype,
    device=device,
)
print("Model loaded successfully.")

# Function to get available Ollama models using CLI command
def get_ollama_models_from_cli():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        output = result.stdout
        models = []
        
        # Parse the CLI output to extract model names
        lines = output.strip().split('\n')
        if len(lines) > 1:  # Skip header line
            for line in lines[1:]:  # Skip the header row
                parts = line.split()
                if parts:  # Make sure line has content
                    models.append(parts[0])  # First column is the model name
        
        if not models:  # Fallback if parsing fails
            return ["llama3", "mistral", "gemma:2b", "phi3:mini"]
        
        print(f"Detected Ollama models: {models}")
        return models
    except Exception as e:
        print(f"Error getting Ollama models via CLI: {str(e)}")
        # Return default models if CLI command fails
        return ["llama3", "mistral", "gemma:2b", "phi3:mini"]

# Function to refresh the model list and update the dropdown
def refresh_models():
    models = get_ollama_models_from_cli()
    return gr.Dropdown(choices=models, value=models[0] if models else "llama3")

# Function to translate text using Ollama
def translate_with_ollama(text, model_name):
    try:
        print(f"Translating with Ollama model: {model_name}")
        
        # Split text into chunks of approximately 15 tokens each
        # A rough approximation is ~4 words per chunk
        words = text.split()
        chunks = []
        current_chunk = []
        word_count = 0
        
        for word in words:
            current_chunk.append(word)
            word_count += 1
            
            # Aim for chunks of ~15 tokens (roughly 4 words)
            if word_count >= 4:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                word_count = 0
        
        # Add any remaining words as the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"Split text into {len(chunks)} chunks for translation")
        
        # Translate each chunk and combine results
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Translating chunk {i+1}/{len(chunks)}: {chunk}")
            prompt = f"Translate the following text to English word for word. DO NOT summarize. If you aren't able to translate just respond [unintelligble]. ONLY return the translated text without ANY explanations or commentary. \n\nTEXT:\n\n{chunk}"
            response = ollama.generate(model=model_name, prompt=prompt)
            translation = response['response'].strip()
            translated_chunks.append(translation)
            print(f"Chunk {i+1} translated: {translation}")
        
        # Combine all translated chunks
        full_translation = ' '.join(translated_chunks)
        print(f"Full translation completed: {len(full_translation)} characters")
        
        return full_translation
    except Exception as e:
        print(f"Translation error with Ollama: {str(e)}")
        return f"Translation failed: {str(e)}"

# Function to convert MP3 to WAV if needed
def convert_to_wav(audio_file):
    """Convert audio file to WAV format if it's an MP3"""
    if audio_file.lower().endswith('.mp3'):
        print(f"Converting MP3 to WAV: {audio_file}")
        wav_file = os.path.splitext(audio_file)[0] + '.wav'
        # If the file exists, use a temporary file instead
        if os.path.exists(wav_file):
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            wav_file = temp_wav
            
        try:
            subprocess.run([
                'ffmpeg', '-i', audio_file, 
                '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
                wav_file
            ], check=True, capture_output=True)
            print(f"Conversion successful: {wav_file}")
            return wav_file
        except subprocess.CalledProcessError as e:
            print(f"Error converting MP3 to WAV: {str(e)}")
            print(f"STDERR: {e.stderr.decode()}")
            return audio_file
    return audio_file

def transcribe(audio_file, language, translate_with_ollama_checkbox, ollama_model, temperature=0.0):
    if audio_file is None:
        return "Please upload an audio file.", ""
    
    try:
        # Convert MP3 to WAV if needed
        processed_audio_file = convert_to_wav(audio_file)
        print(f"Transcribing file: {processed_audio_file} in language: {language}, temperature: {temperature}")
        
        # For long files, we need to use better chunking parameters
        result = whisper(
            processed_audio_file,
            generate_kwargs={
                "language": language,
                "task": "transcribe",
                "temperature": temperature,
                "no_repeat_ngram_size": 3,  # Prevent repetition of 3-grams
                "repetition_penalty": 1.2,  # Penalize repetition
                "max_new_tokens": 445      # Reduced to stay under 448 combined length limit
            },
            chunk_length_s=20,            # Smaller chunks to avoid hallucinations
            stride_length_s=[2, 2],       # Smaller overlap for more precision
            return_timestamps=True,       # Get timestamps to help with debugging
            batch_size=4                  # Smaller batch to process more carefully
        )
        
        transcription = result["text"]
        print(f"Transcription result: {transcription}")
        
        # Check for hallucination patterns (repetitive text)
        repetition_patterns = [
            (r'(\?\s*\w+\?)\s*\1\s*\1\s*\1', '\\1'),  # Repeated questions
            (r'(\b\w{1,3}\b[,\.\s]*)\1{5,}', '\\1'),  # Short words repeated many times
            (r'(\d\s*){10,}', ''),                    # Long sequences of digits
            (r'([^\w\s])\1{5,}', '\\1'),              # Repeated punctuation
            (r'(\b\w+\b)(\s+\1){5,}', '\\1')          # Same word repeated consecutively
        ]
        
        original_length = len(transcription)
        for pattern, replacement in repetition_patterns:
            transcription = re.sub(pattern, replacement, transcription)
        
        if len(transcription) < original_length * 0.7:  # If we removed >30% of the text
            print("Warning: Detected and fixed hallucination patterns in transcription")
            
        # Translate to English if not already in English and translation is requested
        if language != "en" and translate_with_ollama_checkbox:
            print(f"Translating from {language} to English using Ollama model: {ollama_model}")
            try:
                translation = translate_with_ollama(transcription, ollama_model)
                return translation, transcription
            except Exception as e:
                print(f"Translation error: {str(e)}")
                return f"Translation failed: {str(e)}", transcription
        else:
            return transcription, ""
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return f"Error during transcription: {str(e)}", ""
    finally:
        # Clean up temp file if created
        if 'processed_audio_file' in locals() and processed_audio_file != audio_file:
            try:
                if os.path.exists(processed_audio_file):
                    os.unlink(processed_audio_file)
            except Exception as e:
                print(f"Error cleaning up temp file: {str(e)}")

# Initial model list
ollama_models = get_ollama_models_from_cli()
print(f"Available Ollama models: {ollama_models}")

# Gradio UI
with gr.Blocks(title="Speech-to-Text Translator using Whisper Base and Ollama") as iface:
    gr.Markdown("# Speech-to-Text Translator using Whisper Base and Ollama")
    gr.Markdown("Upload an audio file (WAV or MP3) and select the language for transcription. Non-English transcriptions can be translated to English using Ollama. The model supports 96+ languages.")
    
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio file (WAV or MP3)")
        
    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=[
                "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", 
                "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
                "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
                "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
                "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
                "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
                "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
                "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
                "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
                "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
            ],
            value="en", 
            label="Select Language (ISO code)"
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            translate_checkbox = gr.Checkbox(label="Translate to English (for non-English languages)", value=True)
        with gr.Column(scale=1):
            temperature_slider = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                value=0.0, 
                step=0.1, 
                label="Temperature (0.0 for deterministic output)"
            )
    
    with gr.Row():
        ollama_model_dropdown = gr.Dropdown(
            choices=ollama_models,
            value=ollama_models[0] if ollama_models else "llama3",
            label="Select Ollama Model for Translation"
        )
        refresh_button = gr.Button("Refresh Model List")
    
    with gr.Row():
        transcribe_button = gr.Button("Transcribe")
    
    with gr.Row():
        translation_output = gr.Textbox(label="English Translation", lines=10)
    
    with gr.Row():
        original_output = gr.Textbox(label="Original Transcription", lines=10)
    
    # Set up events
    transcribe_button.click(
        fn=transcribe,
        inputs=[audio_input, language_dropdown, translate_checkbox, ollama_model_dropdown, temperature_slider],
        outputs=[translation_output, original_output]
    )
    
    refresh_button.click(
        fn=refresh_models,
        inputs=[],
        outputs=ollama_model_dropdown
    )

print("Starting Gradio web interface...")
iface.launch()