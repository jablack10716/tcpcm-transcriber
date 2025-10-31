"""Export utilities for various transcript formats (SRT, VTT, JSON, JSONL)."""

import json
import logging
from pathlib import Path
from typing import List, Iterable, Dict
from .schemas import Segment, Transcript, Chunk

logger = logging.getLogger(__name__)


def format_timestamp_srt(seconds: float) -> str:
    """
    Format timestamp for SRT format: HH:MM:SS,mmm
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """
    Format timestamp for VTT format: HH:MM:SS.mmm
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export_srt(segments: List[Segment], output_path: str) -> None:
    """
    Export segments to SRT subtitle format.
    
    Args:
        segments: List of transcription segments
        output_path: Output file path
    """
    lines = []
    
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}")
        lines.append(seg.text)
        lines.append("")  # Empty line between entries
    
    output = "\n".join(lines)
    
    Path(output_path).write_text(output, encoding='utf-8')
    logger.info(f"Exported SRT to: {output_path}")


def export_vtt(segments: List[Segment], output_path: str) -> None:
    """
    Export segments to VTT subtitle format.
    
    Args:
        segments: List of transcription segments
        output_path: Output file path
    """
    lines = ["WEBVTT", ""]  # VTT header
    
    for seg in segments:
        lines.append(f"{format_timestamp_vtt(seg.start)} --> {format_timestamp_vtt(seg.end)}")
        lines.append(seg.text)
        lines.append("")  # Empty line between entries
    
    output = "\n".join(lines)
    
    Path(output_path).write_text(output, encoding='utf-8')
    logger.info(f"Exported VTT to: {output_path}")


def export_json(transcript: Transcript, output_path: str) -> None:
    """
    Export full transcript to JSON format.
    
    Args:
        transcript: Transcript object
        output_path: Output file path
    """
    # Convert to dict using Pydantic's model_dump
    data = transcript.model_dump()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported JSON to: {output_path}")


def export_jsonl(chunks: List[Chunk], output_path: str) -> None:
    """
    Export chunks to JSONL format (one JSON object per line).
    
    Args:
        chunks: List of chunks
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            # Convert to dict using Pydantic's model_dump
            data = chunk.model_dump()
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    logger.info(f"Exported JSONL to: {output_path} ({len(chunks)} chunks)")


def export_all(
    transcript: Transcript,
    chunks: List[Chunk],
    output_dir: str,
    base_name: str
) -> dict:
    """
    Export transcript in all formats.
    
    Args:
        transcript: Transcript object
        chunks: List of chunks
        output_dir: Output directory
        base_name: Base name for output files (without extension)
    
    Returns:
        Dictionary mapping format to output path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # Export SRT
    srt_path = output_path / f"{base_name}.srt"
    export_srt(transcript.segments, str(srt_path))
    outputs['srt'] = str(srt_path)
    
    # Export VTT
    vtt_path = output_path / f"{base_name}.vtt"
    export_vtt(transcript.segments, str(vtt_path))
    outputs['vtt'] = str(vtt_path)
    
    # Export JSON
    json_path = output_path / f"{base_name}.json"
    export_json(transcript, str(json_path))
    outputs['json'] = str(json_path)
    
    # Export JSONL
    jsonl_path = output_path / f"{base_name}_chunks.jsonl"
    export_jsonl(chunks, str(jsonl_path))
    outputs['jsonl'] = str(jsonl_path)
    
    logger.info(f"Exported all formats to: {output_dir}")
    return outputs


def export_formats(
    transcript: Transcript,
    chunks: List[Chunk],
    output_dir: str,
    base_name: str,
    formats: Iterable[str]
) -> Dict[str, str]:
    """
    Export transcript in selected formats.

    Args:
        transcript: Transcript object
        chunks: List of chunks
        output_dir: Output directory
        base_name: Base name for output files (without extension)
        formats: Iterable of format strings among {"srt","vtt","json","jsonl"}

    Returns:
        Dictionary mapping format to output path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}

    fmt_set = {f.lower() for f in formats}

    if "srt" in fmt_set:
        srt_path = output_path / f"{base_name}.srt"
        export_srt(transcript.segments, str(srt_path))
        outputs["srt"] = str(srt_path)

    if "vtt" in fmt_set:
        vtt_path = output_path / f"{base_name}.vtt"
        export_vtt(transcript.segments, str(vtt_path))
        outputs["vtt"] = str(vtt_path)

    if "json" in fmt_set:
        json_path = output_path / f"{base_name}.json"
        export_json(transcript, str(json_path))
        outputs["json"] = str(json_path)

    if "jsonl" in fmt_set:
        jsonl_path = output_path / f"{base_name}_chunks.jsonl"
        export_jsonl(chunks, str(jsonl_path))
        outputs["jsonl"] = str(jsonl_path)

    logger.info(f"Exported selected formats ({', '.join(sorted(fmt_set))}) to: {output_dir}")
    return outputs
