"""Tests for export functionality and timecode formatting."""

import pytest
import json
import tempfile
from pathlib import Path
from tcpcm_transcriber.schemas import Segment, Transcript, Chunk
from tcpcm_transcriber.export import (
    format_timestamp_srt,
    format_timestamp_vtt,
    export_srt,
    export_vtt,
    export_json,
    export_jsonl,
)


def test_format_timestamp_srt():
    """Test SRT timestamp formatting."""
    # Test zero
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    
    # Test seconds
    assert format_timestamp_srt(45.5) == "00:00:45,500"
    
    # Test minutes
    assert format_timestamp_srt(125.250) == "00:02:05,250"
    
    # Test hours
    assert format_timestamp_srt(3665.123) == "01:01:05,123"
    
    # Test full time
    assert format_timestamp_srt(3723.456) == "01:02:03,456"


def test_format_timestamp_vtt():
    """Test VTT timestamp formatting."""
    # Test zero
    assert format_timestamp_vtt(0.0) == "00:00:00.000"
    
    # Test seconds
    assert format_timestamp_vtt(45.5) == "00:00:45.500"
    
    # Test minutes
    assert format_timestamp_vtt(125.250) == "00:02:05.250"
    
    # Test hours
    assert format_timestamp_vtt(3665.123) == "01:01:05.123"
    
    # Test full time
    assert format_timestamp_vtt(3723.456) == "01:02:03.456"


def test_export_srt():
    """Test SRT export."""
    segments = [
        Segment(id=0, start=0.0, end=5.0, text="First segment"),
        Segment(id=1, start=5.0, end=10.0, text="Second segment"),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
        output_path = f.name
    
    try:
        export_srt(segments, output_path)
        
        content = Path(output_path).read_text()
        
        # Check format
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:05,000" in content
        assert "First segment" in content
        
        assert "2\n" in content
        assert "00:00:05,000 --> 00:00:10,000" in content
        assert "Second segment" in content
        
    finally:
        Path(output_path).unlink()


def test_export_vtt():
    """Test VTT export."""
    segments = [
        Segment(id=0, start=0.0, end=5.0, text="First segment"),
        Segment(id=1, start=5.0, end=10.0, text="Second segment"),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vtt', delete=False) as f:
        output_path = f.name
    
    try:
        export_vtt(segments, output_path)
        
        content = Path(output_path).read_text()
        
        # Check format
        assert "WEBVTT" in content
        assert "00:00:00.000 --> 00:00:05.000" in content
        assert "First segment" in content
        assert "00:00:05.000 --> 00:00:10.000" in content
        assert "Second segment" in content
        
    finally:
        Path(output_path).unlink()


def test_export_json():
    """Test JSON export."""
    transcript = Transcript(
        segments=[
            Segment(id=0, start=0.0, end=5.0, text="First segment"),
            Segment(id=1, start=5.0, end=10.0, text="Second segment"),
        ],
        language="en",
        duration=10.0
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = f.name
    
    try:
        export_json(transcript, output_path)
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert len(data['segments']) == 2
        assert data['language'] == 'en'
        assert data['duration'] == 10.0
        assert data['segments'][0]['text'] == 'First segment'
        
    finally:
        Path(output_path).unlink()


def test_export_jsonl():
    """Test JSONL export."""
    chunks = [
        Chunk(
            chunk_id=0,
            text="First chunk",
            start=0.0,
            end=5.0,
            segment_ids=[0],
            char_count=11,
            source_file="test.mp4"
        ),
        Chunk(
            chunk_id=1,
            text="Second chunk",
            start=5.0,
            end=10.0,
            segment_ids=[1],
            char_count=12,
            source_file="test.mp4"
        ),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        output_path = f.name
    
    try:
        export_jsonl(chunks, output_path)
        
        lines = Path(output_path).read_text().strip().split('\n')
        
        assert len(lines) == 2
        
        # Parse each line as JSON
        chunk1 = json.loads(lines[0])
        chunk2 = json.loads(lines[1])
        
        assert chunk1['chunk_id'] == 0
        assert chunk1['text'] == 'First chunk'
        assert chunk2['chunk_id'] == 1
        assert chunk2['text'] == 'Second chunk'
        
    finally:
        Path(output_path).unlink()
