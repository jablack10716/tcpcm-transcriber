"""Text chunking utilities for RAG ingestion with timestamp preservation."""

import logging
from typing import List
from .schemas import Segment, Chunk

logger = logging.getLogger(__name__)


class TextChunker:
    """Character-window based text chunker that preserves timestamps."""
    
    def __init__(
        self,
        target_chars: int = 1200,
        overlap_chars: int = 200
    ):
        """
        Initialize text chunker.
        
        Args:
            target_chars: Target character count per chunk
            overlap_chars: Overlap between chunks in characters
        """
        if overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be less than target_chars")
        
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars
        self.stride = target_chars - overlap_chars
    
    def chunk_segments(
        self,
        segments: List[Segment],
        source_file: str = None
    ) -> List[Chunk]:
        """
        Chunk segments into overlapping text chunks.
        
        Args:
            segments: List of transcription segments
            source_file: Source file name for metadata
        
        Returns:
            List of chunks with metadata
        """
        if not segments:
            return []
        
        chunks = []
        chunk_id = 0
        
        # Combine all segments into one text with tracking
        full_text = ""
        char_to_segment = []  # Maps character index to segment
        
        for seg in segments:
            seg_start_char = len(full_text)
            seg_text = seg.text + " "  # Add space between segments
            full_text += seg_text
            
            # Track which segment each character belongs to
            for _ in range(len(seg_text)):
                char_to_segment.append(seg)
        
        full_text = full_text.strip()
        
        # Create chunks with sliding window
        start_char = 0
        
        while start_char < len(full_text):
            end_char = min(start_char + self.target_chars, len(full_text))
            
            # Extract chunk text
            chunk_text = full_text[start_char:end_char].strip()
            
            if not chunk_text:
                break
            
            # Find segments that contributed to this chunk
            chunk_segments = set()
            for i in range(start_char, min(end_char, len(char_to_segment))):
                if i < len(char_to_segment):
                    chunk_segments.add(char_to_segment[i])
            
            chunk_segments = sorted(chunk_segments, key=lambda s: s.id)
            
            if chunk_segments:
                # Get timestamps from first and last segment in chunk
                chunk_start = chunk_segments[0].start
                chunk_end = chunk_segments[-1].end
                segment_ids = [seg.id for seg in chunk_segments]
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start=chunk_start,
                    end=chunk_end,
                    segment_ids=segment_ids,
                    char_count=len(chunk_text),
                    source_file=source_file
                )
                
                chunks.append(chunk)
                chunk_id += 1
            
            # Move window forward
            start_char += self.stride
            
            # Stop if we've reached the end
            if end_char >= len(full_text):
                break
        
        logger.info(f"Created {len(chunks)} chunks from {len(segments)} segments")
        return chunks


def chunk_transcript(
    segments: List[Segment],
    target_chars: int = 1200,
    overlap_chars: int = 200,
    source_file: str = None
) -> List[Chunk]:
    """
    Convenience function to chunk transcript segments.
    
    Args:
        segments: List of transcription segments
        target_chars: Target character count per chunk
        overlap_chars: Overlap between chunks in characters
        source_file: Source file name for metadata
    
    Returns:
        List of chunks with metadata
    """
    chunker = TextChunker(target_chars, overlap_chars)
    return chunker.chunk_segments(segments, source_file)
