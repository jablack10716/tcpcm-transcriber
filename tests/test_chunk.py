"""Tests for text chunking."""

import pytest
from tcpcm_transcriber.schemas import Segment
from tcpcm_transcriber.chunk import TextChunker, chunk_transcript


def create_test_segments():
    """Create test segments for chunking."""
    segments = [
        Segment(id=0, start=0.0, end=5.0, text="This is the first segment."),
        Segment(id=1, start=5.0, end=10.0, text="This is the second segment."),
        Segment(id=2, start=10.0, end=15.0, text="This is the third segment."),
        Segment(id=3, start=15.0, end=20.0, text="This is the fourth segment."),
        Segment(id=4, start=20.0, end=25.0, text="This is the fifth segment."),
    ]
    return segments


def test_chunk_creation():
    """Test that chunks are created correctly."""
    segments = create_test_segments()
    chunker = TextChunker(target_chars=50, overlap_chars=10)
    
    chunks = chunker.chunk_segments(segments)
    
    # Should create multiple chunks
    assert len(chunks) > 0
    
    # Each chunk should have required fields
    for chunk in chunks:
        assert chunk.chunk_id >= 0
        assert chunk.text
        assert chunk.start >= 0
        assert chunk.end > chunk.start
        assert len(chunk.segment_ids) > 0
        assert chunk.char_count > 0


def test_chunk_overlap():
    """Test that chunks have correct overlap."""
    segments = create_test_segments()
    target_chars = 50
    overlap_chars = 10
    
    chunker = TextChunker(target_chars=target_chars, overlap_chars=overlap_chars)
    chunks = chunker.chunk_segments(segments)
    
    if len(chunks) > 1:
        # Check that consecutive chunks have overlap
        # The overlap is approximate due to word boundaries
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            
            # Chunks should have some overlapping content or adjacent segments
            # At minimum, second chunk should start before first chunk ends
            assert chunk2.start <= chunk1.end


def test_chunk_timestamps():
    """Test that chunk timestamps are preserved correctly."""
    segments = create_test_segments()
    chunker = TextChunker(target_chars=50, overlap_chars=10)
    
    chunks = chunker.chunk_segments(segments)
    
    for chunk in chunks:
        # Chunk start should match first segment's start
        first_seg_id = chunk.segment_ids[0]
        first_seg = segments[first_seg_id]
        assert chunk.start == first_seg.start
        
        # Chunk end should match last segment's end
        last_seg_id = chunk.segment_ids[-1]
        last_seg = segments[last_seg_id]
        assert chunk.end == last_seg.end


def test_chunk_segment_ids():
    """Test that segment IDs are tracked correctly."""
    segments = create_test_segments()
    chunker = TextChunker(target_chars=50, overlap_chars=10)
    
    chunks = chunker.chunk_segments(segments)
    
    for chunk in chunks:
        # Should have at least one segment
        assert len(chunk.segment_ids) > 0
        
        # Segment IDs should be in order
        assert chunk.segment_ids == sorted(chunk.segment_ids)
        
        # All segment IDs should be valid
        for seg_id in chunk.segment_ids:
            assert 0 <= seg_id < len(segments)


def test_chunk_character_count():
    """Test that character counts are accurate."""
    segments = create_test_segments()
    chunker = TextChunker(target_chars=50, overlap_chars=10)
    
    chunks = chunker.chunk_segments(segments)
    
    for chunk in chunks:
        assert chunk.char_count == len(chunk.text)


def test_empty_segments():
    """Test handling of empty segment list."""
    chunker = TextChunker(target_chars=50, overlap_chars=10)
    chunks = chunker.chunk_segments([])
    
    assert chunks == []


def test_single_segment():
    """Test chunking with a single segment."""
    segments = [Segment(id=0, start=0.0, end=5.0, text="Short text.")]
    chunker = TextChunker(target_chars=50, overlap_chars=10)
    
    chunks = chunker.chunk_segments(segments)
    
    # Should create at least one chunk
    assert len(chunks) >= 1
    assert chunks[0].text.strip() == "Short text."


def test_target_chars_validation():
    """Test that overlap must be less than target."""
    with pytest.raises(ValueError):
        TextChunker(target_chars=50, overlap_chars=50)
    
    with pytest.raises(ValueError):
        TextChunker(target_chars=50, overlap_chars=60)


def test_convenience_function():
    """Test the convenience function."""
    segments = create_test_segments()
    chunks = chunk_transcript(segments, target_chars=50, overlap_chars=10)
    
    assert len(chunks) > 0


def test_source_file_metadata():
    """Test that source file metadata is included."""
    segments = create_test_segments()
    chunker = TextChunker(target_chars=50, overlap_chars=10)
    
    chunks = chunker.chunk_segments(segments, source_file="test.mp4")
    
    for chunk in chunks:
        assert chunk.source_file == "test.mp4"


def test_long_segments():
    """Test chunking with longer segments."""
    long_text = "A" * 100  # 100 character segment
    segments = [
        Segment(id=0, start=0.0, end=10.0, text=long_text),
        Segment(id=1, start=10.0, end=20.0, text=long_text),
        Segment(id=2, start=20.0, end=30.0, text=long_text),
    ]
    
    chunker = TextChunker(target_chars=150, overlap_chars=30)
    chunks = chunker.chunk_segments(segments)
    
    # Should create multiple chunks
    assert len(chunks) > 1
    
    # Check that chunks respect target size (approximately)
    for chunk in chunks:
        # Chunks should be around target size or less
        assert chunk.char_count <= chunker.target_chars * 1.1  # Allow 10% tolerance
