"""Command-line interface for TcPCM Transcriber."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from .asr import ASREngine
from .media import validate_media_file, probe_media
from .normalize import TextNormalizer
from .chunk import TextChunker
from .export import export_all, export_formats

# Setup logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """TcPCM Transcriber - Transcribe training videos with Whisper."""
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--out', '-o', 'output_dir', 
              type=click.Path(), 
              default='data/output',
              help='Output directory for transcripts')
@click.option('--model', '-m', 
              default='medium',
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model size')
@click.option('--compute-type', 
              type=click.Choice(['int8', 'int8_float16', 'float16', 'float32']),
              help='Compute type (auto-detect if not specified)')
@click.option('--beam-size', 
              type=int, 
              default=5,
              help='Beam size for decoding')
@click.option('--vad/--novad', 
              default=True,
              help='Use VAD filtering')
@click.option('--normalize/--no-normalize', 
              default=True,
              help='Apply text normalization')
@click.option('--glossary', 
              type=click.Path(exists=True),
              help='Path to custom glossary JSON file')
@click.option('--target-chars', 
              type=int, 
              default=1200,
              help='Target characters per chunk')
@click.option('--overlap-chars', 
              type=int, 
              default=200,
              help='Overlap characters between chunks')
@click.option('--language', 
              type=str,
              help='Language code (e.g., "en"). Auto-detect if not specified.')
@click.option('--format', 'formats', multiple=True,
              type=click.Choice(['srt', 'vtt', 'json', 'jsonl', 'all'], case_sensitive=False),
              help='Select one or more export formats. Use multiple --format flags. If omitted or if "all" is included, exports all formats.')
def transcribe(
    input_file: str,
    output_dir: str,
    model: str,
    compute_type: Optional[str],
    beam_size: int,
    vad: bool,
    normalize: bool,
    glossary: Optional[str],
    target_chars: int,
    overlap_chars: int,
    language: Optional[str],
    formats: tuple[str, ...] | None
):
    """Transcribe a single video or audio file."""
    
    try:
        console.print(f"[bold blue]TcPCM Transcriber[/bold blue]")
        console.print(f"Input: {input_file}")
        
        # Validate input
        if not validate_media_file(input_file):
            console.print("[bold red]Error:[/bold red] Invalid input file")
            sys.exit(1)
        
        # Probe media (optional)
        media_info = probe_media(input_file)
        if media_info:
            console.print(f"Duration: {media_info.get('duration', 'unknown')}s")
        
        # Initialize ASR engine
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task(description="Loading Whisper model...", total=None)
            asr = ASREngine(
                model_size=model,
                device=None if compute_type is None else 'cuda' if 'cuda' in compute_type else 'cpu',
                compute_type=compute_type,
                beam_size=beam_size,
                vad_filter=vad
            )
        
        # Transcribe with progress bar
        total_duration = media_info.get('duration') if media_info else None
        
        if total_duration:
            from rich.progress import BarColumn, TimeRemainingColumn
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("{task.fields[segments]} segments"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(
                    "Transcribing...", 
                    total=total_duration,
                    segments=0
                )
                
                def update_progress(current_time: float, segment_count: int):
                    progress.update(task, completed=min(current_time, total_duration), segments=segment_count)
                
                transcript = asr.transcribe(
                    input_file, 
                    language=language,
                    progress_callback=update_progress,
                    total_duration=total_duration
                )
        else:
            console.print("Transcribing... (this may take a while)")
            transcript = asr.transcribe(input_file, language=language)
        
        console.print(f"[green]✓[/green] Transcribed {len(transcript.segments)} segments")
        
        # Normalize text if requested
        if normalize:
            normalizer = TextNormalizer(glossary_path=glossary)
            for seg in transcript.segments:
                seg.text = normalizer.normalize(seg.text)
            console.print("[green]✓[/green] Text normalized")
        
        # Create chunks
        chunker = TextChunker(target_chars=target_chars, overlap_chars=overlap_chars)
        source_file = Path(input_file).name
        chunks = chunker.chunk_segments(transcript.segments, source_file=source_file)
        console.print(f"[green]✓[/green] Created {len(chunks)} RAG chunks")
        
        # Generate output filename
        input_stem = Path(input_file).stem
        # Sanitize filename: remove spaces, convert to lowercase, prefix with tcpcm_
        safe_stem = "tcpcm_" + input_stem.lower().replace(" ", "_")
        # Simplify if it contains "ch" and numbers
        import re
        match = re.search(r'ch\s*(\d+)', input_stem, re.IGNORECASE)
        if match:
            safe_stem = f"tcpcm_ch{match.group(1).zfill(2)}"
        
        # Export selected formats (default: all)
        selected = set((f.lower() for f in (formats or ())))
        if not selected or 'all' in selected:
            outputs = export_all(transcript, chunks, output_dir, safe_stem)
        else:
            outputs = export_formats(transcript, chunks, output_dir, safe_stem, selected)
        
        console.print(f"\n[bold green]Transcription complete![/bold green]")
        console.print(f"Output directory: {output_dir}")
        for fmt, path in outputs.items():
            console.print(f"  {fmt.upper()}: {Path(path).name}")
        
    except Exception as e:
        logger.exception("Transcription failed")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--out', '-o', 'output_dir', 
              type=click.Path(), 
              default='data/output',
              help='Output directory for transcripts')
@click.option('--pattern', 
              default='*.mp4',
              help='File pattern to match (e.g., "*.mp4", "*.wav")')
@click.option('--model', '-m', 
              default='medium',
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model size')
@click.option('--normalize/--no-normalize', 
              default=True,
              help='Apply text normalization')
def batch(
    input_dir: str,
    output_dir: str,
    pattern: str,
    model: str,
    normalize: bool
):
    """Batch transcribe multiple files in a directory."""
    
    try:
        input_path = Path(input_dir)
        files = list(input_path.glob(pattern))
        
        if not files:
            console.print(f"[yellow]No files matching '{pattern}' found in {input_dir}[/yellow]")
            return
        
        console.print(f"[bold blue]TcPCM Transcriber - Batch Mode[/bold blue]")
        console.print(f"Found {len(files)} files to process")
        
        for i, file in enumerate(files, 1):
            console.print(f"\n[bold]Processing {i}/{len(files)}:[/bold] {file.name}")
            
            # Call transcribe for each file
            ctx = click.get_current_context()
            ctx.invoke(
                transcribe,
                input_file=str(file),
                output_dir=output_dir,
                model=model,
                compute_type=None,
                beam_size=5,
                vad=True,
                normalize=normalize,
                glossary=None,
                target_chars=1200,
                overlap_chars=200,
                language=None
            )
        
        console.print(f"\n[bold green]Batch processing complete![/bold green]")
        console.print(f"Processed {len(files)} files")
        
    except Exception as e:
        logger.exception("Batch processing failed")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
