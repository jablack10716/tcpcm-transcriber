"""Media handling utilities with optional ffmpeg probing."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def probe_media(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Probe media file using ffmpeg.
    
    Args:
        file_path: Path to media file
    
    Returns:
        Dictionary with media info or None if probe fails
    """
    try:
        import ffmpeg
        
        probe = ffmpeg.probe(file_path)
        
        # Extract audio and video stream info
        audio_streams = [s for s in probe.get('streams', []) if s['codec_type'] == 'audio']
        video_streams = [s for s in probe.get('streams', []) if s['codec_type'] == 'video']
        
        info = {
            'format': probe.get('format', {}).get('format_name'),
            'duration': float(probe.get('format', {}).get('duration', 0)),
            'has_audio': len(audio_streams) > 0,
            'has_video': len(video_streams) > 0,
            'audio_streams': len(audio_streams),
            'video_streams': len(video_streams),
        }
        
        if audio_streams:
            info['audio_codec'] = audio_streams[0].get('codec_name')
            info['sample_rate'] = audio_streams[0].get('sample_rate')
        
        logger.info(f"Media probe: {info}")
        return info
        
    except ImportError:
        logger.warning("ffmpeg-python not available, skipping media probe")
        return None
    except Exception as e:
        logger.warning(f"Failed to probe media file: {e}")
        return None


def validate_media_file(file_path: str) -> bool:
    """
    Validate that a media file exists and is accessible.
    
    Args:
        file_path: Path to media file
    
    Returns:
        True if file is valid, False otherwise
    """
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False
    
    if not path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return False
    
    if path.stat().st_size == 0:
        logger.error(f"File is empty: {file_path}")
        return False
    
    logger.info(f"Media file validated: {file_path} ({path.stat().st_size} bytes)")
    return True


def get_audio_path(video_path: str) -> str:
    """
    Get audio path from video. 
    For now, just returns the video path as faster-whisper can handle video files directly.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Path to audio (currently just returns video_path)
    """
    # faster-whisper can handle video files directly via ffmpeg
    return video_path
