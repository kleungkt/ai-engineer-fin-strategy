"""
Generator Module
================

Code generation module for AI Strategy Generator.

This module takes parsed NLU output (intent, entities, trading logic)
and generates executable Python strategy code.

Modules:
- code_generator: LLM-based strategy code generation
- template_engine: Pre-built strategy templates for common patterns
- ast_validator: AST-based code validation
- safety_checker: Code safety checks and sandboxed execution
"""

from .code_generator import GeneratedStrategy, generate_strategy, refine_strategy
from .template_engine import STRATEGY_TEMPLATES, render_template, list_templates
from .ast_validator import ValidationResult, validate_code
from .safety_checker import SafetyReport, check_safety, sandbox_execute

__all__ = [
    # 代码生成器
    "GeneratedStrategy",
    "generate_strategy",
    "refine_strategy",
    # 模板引擎
    "STRATEGY_TEMPLATES",
    "render_template",
    "list_templates",
    # AST 验证器
    "ValidationResult",
    "validate_code",
    # 安全检查器
    "SafetyReport",
    "check_safety",
    "sandbox_execute",
]
