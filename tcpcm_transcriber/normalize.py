"""Text normalization with glossary support and filler removal."""

import re
import logging
from typing import Dict, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Common filler words to remove
FILLER_WORDS = [
    "um", "uh", "hmm", "mhm", "uh-huh", "mm-hmm",
    "like", "you know", "i mean", "sort of", "kind of"
]


class TextNormalizer:
    """Text normalizer with glossary-based term mapping and filler removal."""
    
    def __init__(self, glossary_path: str = None, remove_fillers: bool = True):
        """
        Initialize text normalizer.
        
        Args:
            glossary_path: Path to glossary JSON file. Uses default if None.
            remove_fillers: Whether to remove filler words
        """
        self.glossary = self._load_glossary(glossary_path)
        self.remove_fillers = remove_fillers
        
        # Create regex patterns for efficient matching
        # Sort by length (longest first) to match longer phrases first
        glossary_terms = sorted(self.glossary.keys(), key=len, reverse=True)
        self.glossary_pattern = self._create_pattern(glossary_terms)
        
        if remove_fillers:
            self.filler_pattern = self._create_pattern(FILLER_WORDS)
        else:
            self.filler_pattern = None
    
    def _load_glossary(self, glossary_path: str = None) -> Dict[str, str]:
        """Load glossary from JSON file."""
        if glossary_path is None:
            # Use default glossary in package
            default_path = Path(__file__).parent / "glossary_default.json"
            glossary_path = str(default_path)
        
        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                glossary = json.load(f)
            logger.info(f"Loaded glossary with {len(glossary)} terms from {glossary_path}")
            return glossary
        except FileNotFoundError:
            logger.warning(f"Glossary file not found: {glossary_path}, using empty glossary")
            return {}
        except Exception as e:
            logger.error(f"Failed to load glossary: {e}")
            return {}
    
    def _create_pattern(self, terms: List[str]) -> re.Pattern:
        """Create a compiled regex pattern from terms."""
        # Escape special regex characters and join with OR
        escaped_terms = [re.escape(term) for term in terms]
        pattern_str = r'\b(' + '|'.join(escaped_terms) + r')\b'
        return re.compile(pattern_str, re.IGNORECASE)
    
    def normalize(self, text: str) -> str:
        """
        Normalize text using glossary and filler removal.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        # Apply glossary replacements
        if self.glossary:
            text = self._apply_glossary(text)
        
        # Remove filler words
        if self.remove_fillers and self.filler_pattern:
            text = self._remove_fillers(text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _apply_glossary(self, text: str) -> str:
        """Apply glossary term replacements."""
        def replace_func(match):
            matched_text = match.group(0)
            # Find the canonical form (case-insensitive lookup)
            for key, value in self.glossary.items():
                if key.lower() == matched_text.lower():
                    return value
            return matched_text
        
        return self.glossary_pattern.sub(replace_func, text)
    
    def _remove_fillers(self, text: str) -> str:
        """Remove filler words from text."""
        return self.filler_pattern.sub('', text)


def normalize_text(
    text: str,
    glossary_path: str = None,
    remove_fillers: bool = True
) -> str:
    """
    Convenience function to normalize text.
    
    Args:
        text: Input text
        glossary_path: Path to glossary JSON file
        remove_fillers: Whether to remove filler words
    
    Returns:
        Normalized text
    """
    normalizer = TextNormalizer(glossary_path, remove_fillers)
    return normalizer.normalize(text)
