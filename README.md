# TcPCM Transcriber

A CLI tool for transcribing TcPCM training videos using OpenAI's Whisper model via faster-whisper.

## Features

- 🎯 **Accurate Transcription**: Uses Whisper AI for high-quality speech-to-text
- 📝 **Multiple Output Formats**: SRT, VTT, JSON, and JSONL
- 🔍 **Text Normalization**: Glossary-based term mapping and filler word removal
- 📦 **RAG-Ready Chunks**: Generates chunked text with metadata for knowledge ingestion
- 🚀 **GPU Acceleration**: Auto-detects CUDA GPU for faster processing
- 🔧 **Flexible CLI**: Configurable models, batch processing, and more

## Installation

```bash
# Install from source
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

### Prerequisites

- Python 3.10 or higher
- ffmpeg (for audio/video processing)

## Quick Start

### Single File Transcription

```bash
# Basic usage
tcpcm transcribe data/input/Ch\ 01\ Intoduction\ to\ Tool\ Costing.mp4 --out data/output

# With custom model and options
tcpcm transcribe video.mp4 --model large --normalize --out output/
```

### Batch Processing

```bash
# Transcribe all MP4 files in a directory
tcpcm batch data/input --pattern "*.mp4" --out data/output
```

## Output Files

For each input video, the tool generates:

1. **SRT Subtitle File** (`tcpcm_ch01.srt`) - Standard subtitle format
2. **VTT Subtitle File** (`tcpcm_ch01.vtt`) - WebVTT format for web players
3. **JSON Transcript** (`tcpcm_ch01.json`) - Full transcript with segments and metadata
4. **JSONL Chunks** (`tcpcm_ch01_chunks.jsonl`) - RAG-ready chunks with timestamps

### Example Output Structure

```
data/output/
├── tcpcm_ch01.srt
├── tcpcm_ch01.vtt
├── tcpcm_ch01.json
└── tcpcm_ch01_chunks.jsonl
```

## CLI Options

### `tcpcm transcribe` Command

```
Usage: tcpcm transcribe [OPTIONS] INPUT_FILE

Options:
  --out, -o PATH              Output directory (default: data/output)
  --model, -m [tiny|base|small|medium|large]
                              Whisper model size (default: medium)
  --compute-type [int8|int8_float16|float16|float32]
                              Compute type (auto-detect if not specified)
  --beam-size INTEGER         Beam size for decoding (default: 5)
  --vad / --novad            Use VAD filtering (default: enabled)
  --normalize / --no-normalize
                              Apply text normalization (default: enabled)
  --glossary PATH            Path to custom glossary JSON file
  --target-chars INTEGER     Target characters per chunk (default: 1200)
  --overlap-chars INTEGER    Overlap characters between chunks (default: 200)
  --language TEXT            Language code (e.g., "en"). Auto-detect if not specified.
  --help                     Show this message and exit
```

### `tcpcm batch` Command

```
Usage: tcpcm batch [OPTIONS] INPUT_DIR

Options:
  --out, -o PATH              Output directory (default: data/output)
  --pattern TEXT             File pattern to match (default: *.mp4)
  --model, -m [tiny|base|small|medium|large]
                              Whisper model size (default: medium)
  --normalize / --no-normalize
                              Apply text normalization (default: enabled)
  --help                     Show this message and exit
```

## Glossary

The tool includes a default glossary for TcPCM-specific terminology:

```json
{
  "tc pcm": "TcPCM",
  "tcpcm": "TcPCM",
  "teamcenter pcm": "TcPCM",
  "teamcenter product cost management": "TcPCM",
  "tool cost": "tool cost",
  "tool costing": "tool costing"
}
```

### Custom Glossary

Create a JSON file with your term mappings:

```json
{
  "variant1": "Canonical Form",
  "variant2": "Canonical Form"
}
```

Use it with:

```bash
tcpcm transcribe video.mp4 --glossary my_glossary.json
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=tcpcm_transcriber

# Run specific test file
pytest tests/test_chunk.py
```

### Project Structure

```
tcpcm-transcriber/
├── pyproject.toml           # Project configuration and dependencies
├── README.md               # This file
├── tcpcm_transcriber/      # Main package
│   ├── __init__.py
│   ├── cli.py             # CLI commands
│   ├── asr.py             # Whisper ASR wrapper
│   ├── media.py           # Media file handling
│   ├── normalize.py       # Text normalization
│   ├── chunk.py           # Text chunking for RAG
│   ├── export.py          # Output format writers
│   ├── schemas.py         # Pydantic data models
│   └── glossary_default.json  # Default glossary
├── tests/                 # Test suite
│   ├── test_chunk.py
│   ├── test_normalize.py
│   └── test_export.py
└── data/                  # Data directory
    ├── input/            # Input videos
    └── output/           # Generated transcripts
```

## Model Sizes

| Model  | Parameters | English-only | Multilingual | Required VRAM | Relative Speed |
|--------|------------|--------------|--------------|---------------|----------------|
| tiny   | 39 M       | ✓            | ✓            | ~1 GB         | ~32x           |
| base   | 74 M       | ✓            | ✓            | ~1 GB         | ~16x           |
| small  | 244 M      | ✓            | ✓            | ~2 GB         | ~6x            |
| medium | 769 M      | ✓            | ✓            | ~5 GB         | ~2x            |
| large  | 1550 M     |              | ✓            | ~10 GB        | 1x             |

The `medium` model provides a good balance between accuracy and speed.

## Performance Tips

1. **Use GPU**: Ensure CUDA is installed for faster transcription
2. **Adjust Model Size**: Use smaller models for faster processing
3. **Disable VAD**: Use `--novad` if experiencing issues with speech detection
4. **Batch Processing**: Process multiple files together for efficiency

## License

MIT

## Acknowledgments

- Built on [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- Uses OpenAI's [Whisper](https://github.com/openai/whisper) models