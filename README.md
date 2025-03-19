# Audio Transcriber with Whisper and Ollama

A powerful and user-friendly application for transcribing audio files in multiple languages using OpenAI's Whisper model with language translation support via Ollama.

## Features

- **Multilingual Support**: Transcribe audio in 96+ languages
- **Translation**: Automatically translate non-English transcriptions to English locally using Ollama compatible models
- **User-Friendly Interface**: Simple Gradio interface for easy interaction
- **Audio Format Support**: Works with WAV and MP3 files (automatic conversion with ffmpeg)
- **Temperature Control**: Adjust model creativity for different transcription scenarios
- **Anti-Hallucination Measures**: Advanced processing to prevent repetitive outputs

## Requirements

- Python 3.8+
- FFmpeg (included or system-installed)
- Ollama (optional, for translation features)
- CUDA-compatible GPU recommended for faster processing

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/audio-transcriber.git
   cd audio-transcriber
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Install Ollama for translation support:
   - Follow instructions at [Ollama's website](https://ollama.ai/) to install
   - Download at least one model (e.g., `ollama pull llama3`)

## Usage

1. Run the application:
   ```
   python transcribe.py
   ```

2. Access the web interface at http://localhost:7860 in your browser

3. Upload an audio file (WAV or MP3)

4. Select the source language of the audio

5. Choose whether to translate non-English transcriptions to English

6. Adjust temperature (0.0 for deterministic output, higher for more creative transcriptions)

7. Click "Transcribe" to process the audio

## How It Works

The application uses the following pipeline:

1. Audio file is uploaded and converted to WAV format if needed
2. Whisper model processes the audio in optimized chunks
3. Post-processing removes hallucinations and repetitive patterns
4. For non-English audio, Ollama can translate the transcription to English
5. Results are displayed in the interface (original and translated versions)

## Troubleshooting

- **FFmpeg Issues**: The app includes FFmpeg, but you may need to install it separately if conversion fails
- **CUDA Out of Memory**: Reduce batch size in the code if you experience GPU memory issues
- **Ollama Connection Errors**: Ensure Ollama is running before starting the transcriber

## License

MIT License

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [Gradio](https://gradio.app/) for the web interface framework
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for model integration
- [Ollama](https://ollama.ai/) for local language model support 
