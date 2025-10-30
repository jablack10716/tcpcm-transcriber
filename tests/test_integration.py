"""Integration tests for the full transcription pipeline."""

import pytest
import tempfile
import json
from pathlib import Path
from tcpcm_transcriber.schemas import Segment, Transcript
from tcpcm_transcriber.normalize import TextNormalizer
from tcpcm_transcriber.chunk import TextChunker
from tcpcm_transcriber.export import export_all


def create_test_transcript():
    """Create a test transcript for integration testing."""
    segments = [
        Segment(id=0, start=0.0, end=5.0, text="Welcome to tc pcm training."),
        Segment(id=1, start=5.0, end=10.0, text="This covers tool costing basics."),
        Segment(id=2, start=10.0, end=15.0, text="Um, we'll learn about teamcenter pcm features."),
        Segment(id=3, start=15.0, end=20.0, text="You know, tool cost estimation is important."),
        Segment(id=4, start=20.0, end=25.0, text="Let's explore tcpcm in detail."),
    ]
    
    transcript = Transcript(
        segments=segments,
        language="en",
        duration=25.0
    )
    
    return transcript


def test_full_pipeline():
    """Test the complete transcription pipeline."""
    # Create test transcript
    transcript = create_test_transcript()
    
    # Apply normalization
    glossary = {
        "tc pcm": "TcPCM",
        "tcpcm": "TcPCM",
        "teamcenter pcm": "TcPCM",
        "tool cost": "tool cost",
        "tool costing": "tool costing"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(glossary, f)
        glossary_path = f.name
    
    try:
        normalizer = TextNormalizer(glossary_path=glossary_path, remove_fillers=True)
        for seg in transcript.segments:
            seg.text = normalizer.normalize(seg.text)
        
        # Verify normalization worked
        assert "TcPCM" in transcript.segments[0].text
        assert "TcPCM" in transcript.segments[2].text
        assert "TcPCM" in transcript.segments[4].text
        
        # Create chunks
        chunker = TextChunker(target_chars=100, overlap_chars=20)
        chunks = chunker.chunk_segments(transcript.segments, source_file="test.mp4")
        
        # Verify chunks were created
        assert len(chunks) > 0
        assert all(chunk.source_file == "test.mp4" for chunk in chunks)
        
        # Export all formats
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = export_all(transcript, chunks, tmpdir, "test_output")
            
            # Verify all files were created
            assert Path(outputs['srt']).exists()
            assert Path(outputs['vtt']).exists()
            assert Path(outputs['json']).exists()
            assert Path(outputs['jsonl']).exists()
            
            # Verify SRT content
            srt_content = Path(outputs['srt']).read_text()
            assert "TcPCM" in srt_content
            assert "00:00:00,000 --> 00:00:05,000" in srt_content
            
            # Verify VTT content
            vtt_content = Path(outputs['vtt']).read_text()
            assert "WEBVTT" in vtt_content
            assert "TcPCM" in vtt_content
            
            # Verify JSON content
            with open(outputs['json'], 'r') as f:
                json_data = json.load(f)
            assert len(json_data['segments']) == 5
            assert json_data['language'] == 'en'
            
            # Verify JSONL content
            jsonl_lines = Path(outputs['jsonl']).read_text().strip().split('\n')
            assert len(jsonl_lines) == len(chunks)
            
            # Parse first chunk
            first_chunk = json.loads(jsonl_lines[0])
            assert 'chunk_id' in first_chunk
            assert 'text' in first_chunk
            assert 'start' in first_chunk
            assert 'end' in first_chunk
            assert first_chunk['source_file'] == 'test.mp4'
    
    finally:
        Path(glossary_path).unlink()


def test_pipeline_without_normalization():
    """Test pipeline without normalization."""
    transcript = create_test_transcript()
    
    # Create chunks without normalization
    chunker = TextChunker(target_chars=100, overlap_chars=20)
    chunks = chunker.chunk_segments(transcript.segments)
    
    assert len(chunks) > 0
    
    # Export
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = export_all(transcript, chunks, tmpdir, "test_no_norm")
        
        # Verify files exist
        assert all(Path(p).exists() for p in outputs.values())
        
        # Verify original text is preserved (not normalized)
        srt_content = Path(outputs['srt']).read_text()
        assert "tc pcm" in srt_content.lower()  # Original form


def test_empty_pipeline():
    """Test pipeline with empty transcript."""
    transcript = Transcript(segments=[], language="en", duration=0.0)
    
    chunker = TextChunker(target_chars=100, overlap_chars=20)
    chunks = chunker.chunk_segments(transcript.segments)
    
    assert len(chunks) == 0
    
    # Export should still work
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = export_all(transcript, chunks, tmpdir, "test_empty")
        assert all(Path(p).exists() for p in outputs.values())
