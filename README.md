# Speech-to-Text Translator with Whisper and Ollama

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)

A powerful cross-language transcription and translation tool that uses OpenAI's Whisper model for accurate speech recognition and Ollama for local translation processing.

## 📋 Features

- **Multi-Language Transcription**: Support for 96+ languages with Whisper base model
- **Real-Time Translation**: Translate non-English audio to English using local Ollama models
- **Advanced Audio Processing**:
  - Automatic conversion between audio formats
  - Supports WAV and MP3 files
  - Built-in ffmpeg integration
- **Optimized for Long Files**:
  - Smart chunking for processing lengthy audio
  - Overlap management to maintain context
  - Anti-hallucination measures to prevent repetitive output
- **Fine Control**:
  - Temperature adjustment for varying transcription creativity
  - Language selection for accurate source processing
- **Clean User Interface**:
  - Simple Gradio web interface
  - Separate display of original and translated content

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- FFmpeg (included or installed separately)
- Ollama (for translation features)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bitzspider/Speech-To-Text-Translator.git
   cd Speech-To-Text-Translator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama** (for translation):
   - Install Ollama from [ollama.ai](https://ollama.ai)
   - Download at least one language model:
     ```bash
     ollama pull llama3    # Recommended default
     # Other options: mistral, gemma:2b, phi3:mini
     ```

4. **Launch the application**:
   ```bash
   python transcribe.py
   ```

5. **Access the UI**:
   - Open your browser at http://localhost:7860

## 💻 Usage

1. **Upload Audio**: Select any WAV or MP3 file
2. **Choose Language**: Select the source language of your audio
3. **Translation Settings**:
   - Enable/disable translation to English
   - Select which Ollama model to use for translation
4. **Adjust Temperature**:
   - 0.0 for deterministic (consistent) results
   - Higher values for more creative transcription
5. **Start Processing**: Click "Transcribe" and wait for results
6. **View Results**:
   - English translation (if enabled)
   - Original transcription in source language

## ⚙️ How It Works

### Transcription Pipeline

```
Audio Input → Format Conversion → Whisper Model → Chunked Processing → Post-Processing → Output
```

1. **Audio Processing**:
   - Input audio is converted to WAV format if needed
   - Audio is normalized and prepared for transcription

2. **Speech Recognition**:
   - Whisper model processes audio in optimal chunks
   - Each chunk is analyzed with context from adjacent chunks
   - Advanced parameters prevent repetition and hallucination

3. **Translation** (if enabled):
   - Original transcription is divided into small chunks
   - Each chunk is translated individually by Ollama
   - Translations are combined while preserving context
   - Progress is tracked and displayed in console

## 🔧 Customization

- **Change Default Model**: Edit `model_id` in the script to use different Whisper models
- **Adjust Chunking**: Modify `chunk_length_s` and `stride_length_s` for different audio segmentation
- **Translation Customization**: Edit the translation prompt in `translate_with_ollama()` function

## 📊 Performance Tips

- **GPU Acceleration**: Using CUDA dramatically improves transcription speed
- **Memory Usage**: Reduce `batch_size` if experiencing memory issues
- **Long Files**: The chunking mechanism handles files of any length
- **Translation Speed**: Smaller LLMs like Phi-3 Mini are faster for translation

## 🔍 Troubleshooting

- **ffmpeg Issues**: Install ffmpeg manually if audio conversion fails
- **CUDA Errors**: Update your GPU drivers or reduce model size
- **Ollama Connection**: Ensure Ollama service is running before starting transcription
- **Repetitive Output**: Increase `repetition_penalty` in the code if seeing hallucinations

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [Ollama](https://ollama.ai/) - Local LLM for translation
- [Gradio](https://gradio.app/) - UI framework
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Model integration 
