"""
Safety Checker
==============

Safety checks for generated strategy code before execution.

Features:
- Block dangerous patterns: file I/O, network, subprocess, eval, exec, __import__
- Warn about risky patterns: infinite loops, large allocations, deep recursion
- Allow safe libraries: pandas, numpy, math, statistics
- Sandboxed execution with timeout protection
"""

import re
import signal
import traceback
import types
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class SafetyReport(BaseModel):
    """Result of code safety analysis.

    Attributes:
        is_safe: Whether the code is safe for sandboxed execution
        risks: List of risk descriptions (warnings, not blocking)
        blocked_patterns: List of blocked dangerous patterns found
    """

    is_safe: bool = True
    risks: List[str] = Field(default_factory=list)
    blocked_patterns: List[str] = Field(default_factory=list)


# ============================================
# 被阻断的模式 (致命危险)
# ============================================

# 正则模式 - 精确匹配危险操作
BLOCKED_REGEX_PATTERNS: List[tuple] = [
    # 文件 I/O
    (r'\bopen\s*\(', "File I/O: open()"),
    (r'\.write\s*\(', "File I/O: .write()"),
    (r'\.read\s*\(', "File I/O: .read()"),
    (r'\.readlines\s*\(', "File I/O: .readlines()"),
    (r'\.writelines\s*\(', "File I/O: .writelines()"),
    # 网络调用
    (r'\bsocket\b', "Network: socket"),
    (r'\bhttplib\b', "Network: httplib"),
    (r'\burllib\b', "Network: urllib"),
    (r'\brequests\b', "Network: requests"),
    (r'\bhttp\.client\b', "Network: http.client"),
    (r'\bhttp\.server\b', "Network: http.server"),
    (r'\bxmlrpc\b', "Network: xmlrpc"),
    (r'\bftplib\b', "Network: ftplib"),
    (r'\bsmtplib\b', "Network: smtplib"),
    (r'\btelnetlib\b', "Network: telnetlib"),
    # 子进程和系统调用
    (r'\bsubprocess\b', "System: subprocess"),
    (r'\bos\.system\s*\(', "System: os.system()"),
    (r'\bos\.popen\s*\(', "System: os.popen()"),
    (r'\bos\.exec', "System: os.exec*()"),
    (r'\bos\.spawn', "System: os.spawn*()"),
    (r'\bos\.startfile', "System: os.startfile()"),
    (r'\bos\.fork', "System: os.fork()"),
    (r'\bos\.kill', "System: os.kill()"),
    (r'\bos\.remove', "System: os.remove()"),
    (r'\bos\.unlink', "System: os.unlink()"),
    (r'\bos\.rename', "System: os.rename()"),
    (r'\bos\.mkdir', "System: os.mkdir()"),
    (r'\bos\.rmdir', "System: os.rmdir()"),
    (r'\bos\.listdir', "System: os.listdir()"),
    (r'\bos\.walk', "System: os.walk()"),
    (r'\bos\.environ', "System: os.environ"),
    (r'\bos\.getenv', "System: os.getenv()"),
    (r'\bos\.path', "System: os.path"),
    # 动态代码执行
    (r'\bexec\s*\(', "Dynamic: exec()"),
    (r'\beval\s*\(', "Dynamic: eval()"),
    (r'\bcompile\s*\(', "Dynamic: compile()"),
    (r'\b__import__\s*\(', "Dynamic: __import__()"),
    (r'\bimportlib\b', "Dynamic: importlib"),
    # 序列化 (可能导致代码注入)
    (r'\bpickle\b', "Serialization: pickle"),
    (r'\bshelve\b', "Serialization: shelve"),
    (r'\bmarshal\b', "Serialization: marshal"),
    # 反射 (绕过安全限制)
    (r'\bglobals\s*\(', "Reflection: globals()"),
    (r'\blocals\s*\(', "Reflection: locals()"),
    (r'\bvars\s*\(', "Reflection: vars()"),
    (r'\bgetattr\s*\(', "Reflection: getattr()"),
    (r'\bsetattr\s*\(', "Reflection: setattr()"),
    (r'\bdelattr\s*\(', "Reflection: delattr()"),
    (r'\bhasattr\s*\(', "Reflection: hasattr()"),
    # 多线程/进程
    (r'\bthreading\b', "Concurrency: threading"),
    (r'\bmultiprocessing\b', "Concurrency: multiprocessing"),
    (r'\bconcurrent\b', "Concurrency: concurrent"),
    (r'\basyncio\b', "Concurrency: asyncio"),
    # 系统退出
    (r'\bexit\s*\(', "System: exit()"),
    (r'\bquit\s*\(', "System: quit()"),
    (r'\b_breakpoint\s*\(', "System: breakpoint()"),
    # 内置受限函数
    (r'\binput\s*\(', "Interactive: input()"),
    # SQL 注入
    (r'\bsqlite3\b', "Database: sqlite3"),
    # 路径操作
    (r'\bpathlib\b', "Filesystem: pathlib"),
    (r'\bglob\b', "Filesystem: glob"),
    (r'\bshutil\b', "Filesystem: shutil"),
    (r'\btempfile\b', "Filesystem: tempfile"),
]

# ============================================
# 风险警告模式 (非致命，仅警告)
# ============================================

WARNING_REGEX_PATTERNS: List[tuple] = [
    (r'while\s+True', "Potential infinite loop: while True"),
    (r'while\s+1', "Potential infinite loop: while 1"),
    (r'for\s+.*\s+in\s+range\s*\(\s*\d{6,}\s*\)', "Large loop: range with 100k+ iterations"),
    (r'\*\*\s*\d{4,}', "Large exponentiation"),
    (r'\[\s*0\s*\]\s*\*\s*\d{6,}', "Large list allocation"),
    (r'\bpd\.DataFrame\s*\(\s*\{.*\}\s*\)', "Dynamic DataFrame creation"),
    (r'\.set_value\b', "Deprecated method: .set_value()"),
    (r'\.ix\b', "Deprecated accessor: .ix"),
    (r'from\s+__future__\s+import\s+annotations', "Future annotations may cause issues"),
]

# ============================================
# 允许的模块列表
# ============================================

ALLOWED_MODULES: Set[str] = {
    # 数据处理
    "pandas",
    "pd",
    "numpy",
    "np",
    # 标准库 - 数学
    "math",
    "statistics",
    "decimal",
    "fractions",
    "random",
    # 标准库 - 数据结构
    "collections",
    "itertools",
    "functools",
    "operator",
    "copy",
    "enum",
    "dataclasses",
    "typing",
    # 标准库 - 日期时间
    "datetime",
    "time",
    "calendar",
    # 标准库 - 字符串
    "string",
    "re",
    "textwrap",
    "unicodedata",
    # 标准库 - 数值
    "cmath",
    # 标准库 - 其他安全模块
    "json",
    "csv",
    "bisect",
    "heapq",
    "array",
    "weakref",
    "contextlib",
    "abc",
    "warnings",
    "copyreg",
    # 第三方安全库
    "scipy",
    "sklearn",
    "ta",           # TA-Lib (technical analysis)
    "talib",
    "ta_lib",
}


def check_safety(code: str) -> SafetyReport:
    """Check code for safety violations and risky patterns.

    Performs regex-based pattern matching to detect:
    - Blocked patterns: file I/O, network, subprocess, eval, exec, etc.
    - Risk patterns: infinite loops, large allocations, etc.

    Args:
        code: Python source code string to check

    Returns:
        SafetyReport with is_safe, risks, and blocked_patterns

    Example:
        >>> report = check_safety("import pandas as pd\\nclass S: pass")
        >>> assert report.is_safe
        >>> report = check_safety("import os\\nos.system('rm -rf /')")
        >>> assert not report.is_safe
    """
    report = SafetyReport()
    blocked: Set[str] = set()
    risks: Set[str] = set()

    lines = code.split("\n")

    # 检查被阻断的模式
    for line_num, line in enumerate(lines, 1):
        # 跳过注释行
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for pattern, description in BLOCKED_REGEX_PATTERNS:
            if re.search(pattern, line):
                blocked.add(f"{description} (line {line_num})")

    # 检查警告模式
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for pattern, description in WARNING_REGEX_PATTERNS:
            if re.search(pattern, line):
                risks.add(f"{description} (line {line_num})")

    # 检查 import 语句中的危险模块
    import_pattern = re.compile(r'^\s*(?:from\s+(\w+)|import\s+(\w+))', re.MULTILINE)
    for match in import_pattern.finditer(code):
        module_name = match.group(1) or match.group(2)
        if module_name and module_name not in ALLOWED_MODULES:
            # 检查是否在阻断列表中
            is_blocked = False
            for _, desc in BLOCKED_REGEX_PATTERNS:
                if module_name in desc.lower() or module_name == desc.split(":")[1].strip().split(".")[0].split("(")[0]:
                    is_blocked = True
                    break
            if not is_blocked:
                risks.add(f"Unlisted module import: '{module_name}'")

    report.blocked_patterns = sorted(blocked)
    report.risks = sorted(risks)
    report.is_safe = len(blocked) == 0

    return report


class _TimeoutError(Exception):
    """内部超时异常"""
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    """信号处理器：超时时抛出异常"""
    raise _TimeoutError("代码执行超时")


def sandbox_execute(
    code: str,
    data: pd.DataFrame,
    timeout: int = 30,
) -> pd.DataFrame:
    """Execute generated strategy code in a restricted namespace.

    Runs the generated code in a sandbox with:
    - Restricted builtins (no exec, eval, open, etc.)
    - Only safe modules available (pandas, numpy, math, etc.)
    - Timeout protection to prevent infinite loops
    - Input/output validation

    Args:
        code: Python source code defining a strategy class
        data: Input DataFrame with OHLCV data
        timeout: Maximum execution time in seconds (default: 30)

    Returns:
        DataFrame with added 'signal' column from generate_signals()

    Raises:
        ValueError: If code doesn't define a valid strategy class
        TimeoutError: If execution exceeds the timeout
        RuntimeError: If code execution fails

    Example:
        >>> code = '''
        ... import pandas as pd
        ... class MyStrategy:
        ...     def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        ...         df = data.copy()
        ...         df["signal"] = 0
        ...         return df
        ... '''
        >>> result = sandbox_execute(code, sample_df, timeout=10)
        >>> assert "signal" in result.columns
    """
    # 安全检查
    report = check_safety(code)
    if not report.is_safe:
        raise ValueError(
            f"代码安全检查未通过:\n"
            f"被阻断的模式: {report.blocked_patterns}"
        )

    # 构建受限的命名空间
    restricted_builtins = _build_restricted_builtins()

    sandbox_globals: Dict[str, Any] = {
        "__builtins__": restricted_builtins,
        "__name__": "__sandbox__",
        "__doc__": None,
        # 安全的标准库
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
    }

    # 尝试导入可选的安全库
    try:
        import math
        sandbox_globals["math"] = math
    except ImportError:
        pass

    try:
        import statistics
        sandbox_globals["statistics"] = statistics
    except ImportError:
        pass

    try:
        import collections
        sandbox_globals["collections"] = collections
    except ImportError:
        pass

    try:
        import copy
        sandbox_globals["copy"] = copy
    except ImportError:
        pass

    try:
        import json
        sandbox_globals["json"] = json
    except ImportError:
        pass

    # 设置超时 (仅在Unix系统上)
    old_handler = None
    use_signal = hasattr(signal, "SIGALRM")
    if use_signal:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        # 执行代码 (定义类)
        exec(compile(code, "<strategy>", "exec"), sandbox_globals)  # noqa: S102

        # 查找策略类
        strategy_class = None
        for obj in sandbox_globals.values():
            if (
                isinstance(obj, type)
                and hasattr(obj, "generate_signals")
                and obj.__name__ != "object"
            ):
                strategy_class = obj
                break

        if strategy_class is None:
            raise ValueError(
                "代码中未找到包含 generate_signals 方法的策略类"
            )

        # 实例化策略
        strategy_instance = strategy_class()

        # 执行 generate_signals
        result = strategy_instance.generate_signals(data)

        # 验证结果
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"generate_signals 返回了 {type(result).__name__}，期望 DataFrame"
            )

        if "signal" not in result.columns:
            raise ValueError(
                "generate_signals 返回的 DataFrame 中缺少 'signal' 列"
            )

        return result

    except _TimeoutError:
        raise TimeoutError(
            f"代码执行超时 ({timeout}秒)，可能存在无限循环"
        )
    except (ValueError, TypeError):
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"策略代码执行失败: {e}\n{tb}") from e
    finally:
        # 恢复信号处理
        if use_signal and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def _build_restricted_builtins() -> Dict[str, Any]:
    """构建受限的 builtins 字典。

    移除所有危险的内置函数，保留安全的数学和类型函数。

    Returns:
        受限的 builtins 字典
    """
    # 安全的内置函数
    safe_builtins = {
        # 类型转换
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "str": str,
        "bytes": bytes,
        "bytearray": bytearray,
        # 集合类型
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        # 数学函数
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "divmod": divmod,
        # 迭代和序列
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "reversed": reversed,
        "sorted": sorted,
        "len": len,
        "iter": iter,
        "next": next,
        # 类型检查
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": type,
        "id": id,
        # 字符串表示
        "repr": repr,
        "ascii": ascii,
        "format": format,
        "chr": chr,
        "ord": ord,
        "bin": bin,
        "oct": oct,
        "hex": hex,
        # 逻辑
        "all": all,
        "any": any,
        # 杂项
        "print": print,
        "hash": hash,
        "callable": callable,
        "slice": slice,
        "property": property,
        "staticmethod": staticmethod,
        "classmethod": classmethod,
        "super": super,
        "object": object,
        "staticmethod": staticmethod,
        # 内置异常
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
        "ZeroDivisionError": ZeroDivisionError,
        "OverflowError": OverflowError,
        "ArithmeticError": ArithmeticError,
        "NotImplementedError": NotImplementedError,
        "NameError": NameError,
    }

    return safe_builtins
