"""
AST Validator
=============

Validate generated strategy code using Python's AST module.

Checks:
- Valid Python syntax
- Class definition exists
- generate_signals method exists
- No dangerous operations (exec, eval, os, subprocess, etc.)
- Required imports (pandas)
"""

import ast
from typing import List, Optional

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of AST-based code validation.

    Attributes:
        is_valid: Whether the code passed all validation checks
        errors: List of critical errors that prevent execution
        warnings: List of non-critical issues
        class_name: Name of the strategy class found (if any)
        has_generate_signals: Whether the code has a generate_signals method
    """

    is_valid: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    class_name: Optional[str] = None
    has_generate_signals: bool = False


# 危险的函数/模块列表
DANGEROUS_FUNCTIONS = frozenset({
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "open",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "help",
    "license",
    "credits",
    "copyright",
})

DANGEROUS_MODULES = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "urllib",
    "requests",
    "ftplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
    "ctypes",
    "importlib",
    "pickle",
    "shelve",
    "marshal",
    "dbm",
    "sqlite3",
    "pathlib",
    "io",
    "tempfile",
    "glob",
    "fnmatch",
    "linecache",
    "tokenize",
    "ast",
    "symtable",
    "code",
    "codeop",
    "compileall",
    "dis",
    "inspect",
    "site",
    "pydoc",
    "threading",
    "multiprocessing",
    "concurrent",
    "asyncio",
    "signal",
    "mmap",
    "winreg",
    "msilib",
    "nis",
    "nntplib",
    "poplib",
    "imaplib",
})

# 被阻断的模式字符串 (用于正则检查)
BLOCKED_PATTERNS = frozenset({
    "os.system",
    "os.popen",
    "os.exec",
    "os.spawn",
    "os.startfile",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "subprocess.check_output",
    "subprocess.check_call",
    "__import__",
    "importlib.import_module",
    "open(",
    ".write(",
    ".read(",
    "exec(",
    "eval(",
    "compile(",
})


class _ASTChecker(ast.NodeVisitor):
    """AST节点访问器，用于检查代码安全性与结构。"""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.class_name: Optional[str] = None
        self.has_generate_signals: bool = False
        self.has_pandas_import: bool = False
        self.imported_modules: List[str] = []
        self._current_class: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """检查类定义"""
        if self.class_name is not None:
            self.warnings.append(
                f"Multiple class definitions found; using first: '{self.class_name}'"
            )
        else:
            self.class_name = node.name

        old_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """检查函数/方法定义"""
        if self._current_class and node.name == "generate_signals":
            self.has_generate_signals = True
            # 检查是否有 self 和 data 参数
            args = node.args
            if len(args.args) < 2:
                self.warnings.append(
                    "generate_signals should accept (self, data) parameters"
                )
            # 检查是否有返回类型注解
            if node.returns is None:
                self.warnings.append(
                    "generate_signals missing return type annotation"
                )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """异步函数同 visit_FunctionDef"""
        if self._current_class and node.name == "generate_signals":
            self.has_generate_signals = True
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """检查 import 语句"""
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            self.imported_modules.append(module_name)
            if module_name in DANGEROUS_MODULES:
                self.errors.append(
                    f"Dangerous import detected: '{alias.name}' (line {node.lineno})"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """检查 from ... import 语句"""
        if node.module:
            module_name = node.module.split(".")[0]
            self.imported_modules.append(module_name)

            if module_name in DANGEROUS_MODULES:
                self.errors.append(
                    f"Dangerous import from '{node.module}' (line {node.lineno})"
                )

            if module_name == "pandas":
                self.has_pandas_import = True

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """检查函数调用"""
        func_name = self._get_call_name(node)
        if func_name in DANGEROUS_FUNCTIONS:
            self.errors.append(
                f"Dangerous function call: '{func_name}()' (line {node.lineno})"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """检查属性访问 (如 os.system)"""
        attr_chain = self._get_attribute_chain(node)
        if attr_chain in BLOCKED_PATTERNS:
            self.errors.append(
                f"Blocked pattern: '{attr_chain}' (line {node.lineno})"
            )
        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> str:
        """获取函数调用的名称"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """获取属性访问链 (如 'os.system')"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value  # type: ignore[assignment]
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return ".".join(parts)


def validate_code(code: str) -> ValidationResult:
    """Validate generated strategy code using Python AST.

    Performs the following checks:
    1. Syntax validation via ast.parse
    2. Class definition exists
    3. generate_signals method exists in the class
    4. No dangerous operations (exec, eval, os, subprocess, etc.)
    5. Required imports (pandas)
    6. No blocked patterns (file I/O, network, etc.)

    Args:
        code: Python source code string to validate

    Returns:
        ValidationResult with is_valid, errors, warnings,
        class_name, and has_generate_signals fields

    Example:
        >>> result = validate_code('''
        ... import pandas as pd
        ... class MyStrategy:
        ...     def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        ...         return data
        ... ''')
        >>> assert result.is_valid
        >>> assert result.class_name == "MyStrategy"
    """
    result = ValidationResult()

    # Step 1: 语法检查
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result.is_valid = False
        result.errors.append(f"Syntax error: {e.msg} (line {e.lineno})")
        return result

    # Step 2-6: AST 深度检查
    checker = _ASTChecker()
    checker.visit(tree)

    # 结构检查
    if checker.class_name is None:
        result.errors.append("No class definition found in generated code")
    else:
        result.class_name = checker.class_name

    if not checker.has_generate_signals:
        result.errors.append(
            "No 'generate_signals' method found in strategy class"
        )

    result.has_generate_signals = checker.has_generate_signals

    # 依赖检查
    if not checker.has_pandas_import:
        result.warnings.append(
            "pandas import not detected; strategy may fail at runtime"
        )

    # 汇总错误和警告
    result.errors.extend(checker.errors)
    result.warnings.extend(checker.warnings)

    # 有error则标记为无效
    if result.errors:
        result.is_valid = False

    return result
