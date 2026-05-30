"""
LLM Utilities Module
====================

This module provides utilities for working with Large Language Models (LLMs)
in financial analysis and natural language processing tasks.

Supported features:
- Prompt templates for common financial NLP tasks
- Response parsing utilities
- LLM client wrappers (placeholder for various providers)

Usage:
    from shared.llm_utils import get_prompt_template, analyze_sentiment
    from shared.llm_utils.prompt_templates import PromptTemplates
"""

from .prompt_templates import (
    PromptTemplates,
    get_prompt_template,
    format_prompt,
)

__all__ = [
    "PromptTemplates",
    "get_prompt_template",
    "format_prompt",
]
