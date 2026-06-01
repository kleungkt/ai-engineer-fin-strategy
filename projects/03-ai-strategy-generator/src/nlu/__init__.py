"""
NLU (Natural Language Understanding) module for AI Strategy Generator.

Provides intent classification, entity extraction, and logic parsing for
natural language trading strategy descriptions in both Chinese and English.
"""

from .entity_extractor import ExtractionResult, TradingEntity, extract_entities
from .intent_classifier import StrategyIntent, classify_intent
from .logic_parser import Condition, TradingLogic, parse_logic

__all__ = [
    # Intent classification
    "StrategyIntent",
    "classify_intent",
    # Entity extraction
    "TradingEntity",
    "ExtractionResult",
    "extract_entities",
    # Logic parsing
    "Condition",
    "TradingLogic",
    "parse_logic",
]
