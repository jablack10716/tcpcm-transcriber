"""Tests for text normalization."""

import pytest
import json
import tempfile
from pathlib import Path
from tcpcm_transcriber.normalize import TextNormalizer, normalize_text


def test_glossary_mapping():
    """Test that glossary terms are correctly mapped."""
    # Create a temporary glossary
    glossary = {
        "tc pcm": "TcPCM",
        "tcpcm": "TcPCM",
        "tool cost": "tool cost",
        "tool costing": "tool costing"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(glossary, f)
        glossary_path = f.name
    
    try:
        normalizer = TextNormalizer(glossary_path=glossary_path, remove_fillers=False)
        
        # Test various forms
        assert normalizer.normalize("tc pcm is great") == "TcPCM is great"
        assert normalizer.normalize("TCPCM is great") == "TcPCM is great"
        assert normalizer.normalize("TC PCM is great") == "TcPCM is great"
        assert normalizer.normalize("tool costing methods") == "tool costing methods"
        
    finally:
        Path(glossary_path).unlink()


def test_filler_removal():
    """Test that filler words are removed."""
    normalizer = TextNormalizer(glossary_path=None, remove_fillers=True)
    
    # Test filler removal
    text = "um so like you know this is a test"
    normalized = normalizer.normalize(text)
    
    # Fillers should be removed
    assert "um" not in normalized.lower()
    assert "like" not in normalized.lower()
    assert "you know" not in normalized.lower()
    assert "test" in normalized


def test_no_filler_removal():
    """Test that fillers are preserved when removal is disabled."""
    normalizer = TextNormalizer(glossary_path=None, remove_fillers=False)
    
    text = "um this is a test"
    normalized = normalizer.normalize(text)
    
    # Fillers should be preserved
    assert "um" in normalized.lower()


def test_whitespace_cleanup():
    """Test that extra whitespace is cleaned up."""
    normalizer = TextNormalizer(glossary_path=None, remove_fillers=False)
    
    text = "this  has   extra    spaces"
    normalized = normalizer.normalize(text)
    
    # Should have single spaces only
    assert "  " not in normalized
    assert normalized == "this has extra spaces"


def test_empty_text():
    """Test handling of empty text."""
    normalizer = TextNormalizer(glossary_path=None)
    
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("   ") == ""


def test_convenience_function():
    """Test the convenience function."""
    text = "tc pcm is great"
    
    # Create temp glossary
    glossary = {"tc pcm": "TcPCM"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(glossary, f)
        glossary_path = f.name
    
    try:
        normalized = normalize_text(text, glossary_path=glossary_path)
        assert "TcPCM" in normalized
    finally:
        Path(glossary_path).unlink()


def test_case_preservation_in_replacement():
    """Test that replacements preserve the correct case."""
    glossary = {"tcpcm": "TcPCM"}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(glossary, f)
        glossary_path = f.name
    
    try:
        normalizer = TextNormalizer(glossary_path=glossary_path, remove_fillers=False)
        
        # Should replace with canonical form regardless of input case
        assert normalizer.normalize("tcpcm") == "TcPCM"
        assert normalizer.normalize("TCPCM") == "TcPCM"
        assert normalizer.normalize("TcPcM") == "TcPCM"
        
    finally:
        Path(glossary_path).unlink()
