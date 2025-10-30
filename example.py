#!/usr/bin/env python
"""
Example script showing how to use tcpcm_transcriber programmatically.
"""

from pathlib import Path
from tcpcm_transcriber.schemas import Segment, Transcript
from tcpcm_transcriber.normalize import TextNormalizer
from tcpcm_transcriber.chunk import TextChunker
from tcpcm_transcriber.export import export_all


def main():
    """Demonstrate programmatic usage of tcpcm_transcriber."""
    
    # Example: Create a transcript (normally this comes from ASR)
    segments = [
        Segment(id=0, start=0.0, end=5.0, text="Welcome to tc pcm training."),
        Segment(id=1, start=5.0, end=10.0, text="This covers tool costing basics."),
        Segment(id=2, start=10.0, end=15.0, text="We'll learn about teamcenter pcm features."),
    ]
    
    transcript = Transcript(
        segments=segments,
        language="en",
        duration=15.0
    )
    
    # Normalize text
    normalizer = TextNormalizer(remove_fillers=True)
    for seg in transcript.segments:
        seg.text = normalizer.normalize(seg.text)
    
    print("Normalized segments:")
    for seg in transcript.segments:
        print(f"  {seg.start:.1f}s - {seg.end:.1f}s: {seg.text}")
    
    # Create chunks for RAG
    chunker = TextChunker(target_chars=1200, overlap_chars=200)
    chunks = chunker.chunk_segments(transcript.segments, source_file="example.mp4")
    
    print(f"\nCreated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  Chunk {chunk.chunk_id}: {chunk.char_count} chars, "
              f"{chunk.start:.1f}s - {chunk.end:.1f}s")
    
    # Export all formats
    output_dir = Path(__file__).parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = export_all(transcript, chunks, str(output_dir), "example")
    
    print("\nExported files:")
    for fmt, path in outputs.items():
        print(f"  {fmt.upper()}: {path}")


if __name__ == "__main__":
    main()
