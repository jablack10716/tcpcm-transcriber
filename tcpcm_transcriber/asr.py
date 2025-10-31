"""ASR (Automatic Speech Recognition) module using faster-whisper."""

import logging
from typing import List, Optional, Callable
from faster_whisper import WhisperModel
from .schemas import Segment, Transcript

logger = logging.getLogger(__name__)


def detect_device() -> tuple[str, str]:
    """
    Auto-detect the best device and compute type.
    
    Returns:
        Tuple of (device, compute_type)
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected, using GPU with int8_float16")
            return "cuda", "int8_float16"
    except ImportError:
        pass
    
    logger.info("No GPU detected, using CPU with int8")
    return "cpu", "int8"


class ASREngine:
    """Wrapper around faster-whisper WhisperModel."""
    
    def __init__(
        self,
        model_size: str = "medium",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        """
        Initialize ASR engine.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda). Auto-detect if None.
            compute_type: Compute type. Auto-detect if None.
            beam_size: Beam size for decoding
            vad_filter: Whether to use VAD filtering
        """
        if device is None or compute_type is None:
            auto_device, auto_compute = detect_device()
            device = device or auto_device
            compute_type = compute_type or auto_compute
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        
        logger.info(f"Loading Whisper model: {model_size} on {device} with {compute_type}")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[float, int], None]] = None,
        total_duration: Optional[float] = None
    ) -> Transcript:
        """
        Transcribe an audio/video file.
        
        Args:
            audio_path: Path to audio or video file
            language: Language code (e.g., 'en'). Auto-detect if None.
            progress_callback: Optional callback function(current_time, segment_count)
            total_duration: Total duration in seconds (for progress estimation)
        
        Returns:
            Transcript object with segments
        """
        logger.info(f"Transcribing: {audio_path}")
        
        segments_iter, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        
        segments = []
        for i, segment in enumerate(segments_iter):
            seg = Segment(
                id=i,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip()
            )
            segments.append(seg)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(seg.end, len(segments))
        
        # Calculate total duration from last segment
        duration = segments[-1].end if segments else 0.0
        
        transcript = Transcript(
            segments=segments,
            language=info.language,
            duration=duration
        )
        
        logger.info(f"Transcription complete: {len(segments)} segments, "
                   f"language={info.language}, duration={duration:.2f}s")
        
        return transcript
