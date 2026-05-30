"""
Prompt Templates Module
=======================

This module provides common prompt templates for financial NLP tasks using
Large Language Models (LLMs).

Supported tasks:
- Sentiment Analysis: Analyze sentiment of financial news/reports
- Summarization: Summarize financial documents
- Entity Extraction: Extract financial entities (companies, metrics, etc.)
- Question Answering: Answer questions about financial data
- Report Generation: Generate financial analysis reports

All templates support variable substitution using Python string formatting.
"""

from typing import Dict, Any, Optional, List
from string import Template


class PromptTemplates:
    """
    Collection of prompt templates for financial NLP tasks.

    This class provides a centralized repository of prompt templates
    that can be used with various LLM providers.
    """

    # ============================================
    # 情感分析模板
    # ============================================

    SENTIMENT_ANALYSIS = Template("""
你是一个专业的金融分析师。请分析以下文本的情感倾向。

文本:
$text

请按照以下JSON格式返回分析结果:
{
    "sentiment": "positive/negative/neutral",
    "confidence": 0.0-1.0,
    "keywords": ["关键词1", "关键词2"],
    "reasoning": "分析原因"
}
""")

    STOCK_SENTIMENT = Template("""
你是一个股票市场分析师。请分析以下关于 $stock_name ($stock_code) 的新闻/评论的情感。

文本:
$text

请从以下维度分析:
1. 整体情感: positive/negative/neutral
2. 对股价的可能影响: bullish/bearish/neutral
3. 关键影响因素
4. 置信度 (0-100%)

请以JSON格式返回。
""")

    # ============================================
    # 文本摘要模板
    # ============================================

    FINANCIAL_SUMMARY = Template("""
你是一个金融文档分析专家。请总结以下金融文档的核心内容。

文档:
$text

请提供:
1. 核心摘要 (100字以内)
2. 关键数据点
3. 主要结论
4. 潜在风险提示
""")

    EARNINGS_SUMMARY = Template("""
请分析以下财报/业绩公告，提取关键财务指标和业绩亮点。

文档:
$text

请提取:
1. 营收数据及同比变化
2. 净利润及同比变化
3. 毛利率/净利率变化
4. 业务亮点
5. 风险提示
6. 管理层展望
""")

    # ============================================
    # 实体提取模板
    # ============================================

    ENTITY_EXTRACTION = Template("""
请从以下金融文本中提取关键实体。

文本:
$text

请提取以下类型的实体:
1. 公司名称 (company)
2. 人物 (person)
3. 财务指标 (metric)
4. 金额 (amount)
5. 日期 (date)
6. 产品/服务 (product)

以JSON格式返回:
{
    "entities": [
        {"text": "实体文本", "type": "实体类型", "normalized": "标准化值"}
    ]
}
""")

    # ============================================
    # 问答模板
    # ============================================

    FINANCIAL_QA = Template("""
你是一个专业的金融分析师。请根据提供的上下文回答问题。

上下文信息:
$context

问题:
$question

请提供:
1. 直接答案
2. 数据支持
3. 分析依据
4. 注意事项/免责声明
""")

    STOCK_ANALYSIS_QA = Template("""
你是一个资深股票分析师。请分析以下股票相关问题。

股票信息:
$stock_info

历史数据:
$historical_data

技术指标:
$technical_indicators

问题:
$question

请从基本面和技术面两个角度进行分析，给出你的观点。
""")

    # ============================================
    # 报告生成模板
    # ============================================

    STOCK_REPORT = Template("""
请生成一份关于 $stock_name ($stock_code) 的投资分析报告。

基础数据:
$stock_data

技术指标:
$technical_data

新闻/公告:
$news

报告结构:
1. 公司概况
2. 近期表现回顾
3. 技术面分析
4. 基本面分析
5. 行业对比
6. 风险提示
7. 投资建议

请以专业、客观的语气撰写，字数约 $word_count 字。
""")

    MARKET_DAILY_REPORT = Template("""
请生成今日A股市场日报。

市场数据:
$market_data

热门板块:
$hot_sectors

重要新闻:
$news

报告结构:
1. 市场概况 (指数表现)
2. 板块分析
3. 资金流向
4. 热点回顾
5. 明日展望

语气专业，字数约500字。
""")

    # ============================================
    # 数据分析模板
    # ============================================

    DATA_INTERPRETATION = Template("""
请解读以下金融数据。

数据:
$data

数据说明:
$description

请分析:
1. 数据的核心含义
2. 与历史数据的对比
3. 异常值识别
4. 趋势判断
5. 投资启示
""")

    # ============================================
    # 策略分析模板
    # ============================================

    TRADING_STRATEGY = Template("""
你是一个量化交易策略分析师。请评估以下交易策略。

策略描述:
$strategy_description

历史回测结果:
$backtest_results

请评估:
1. 策略的逻辑合理性
2. 风险收益特征
3. 适用市场环境
4. 潜在风险
5. 改进建议
""")


def get_prompt_template(template_name: str) -> Template:
    """
    Get a prompt template by name.

    Args:
        template_name: Name of the template (e.g., 'SENTIMENT_ANALYSIS')

    Returns:
        Template: The requested prompt template

    Raises:
        AttributeError: If template_name doesn't exist

    Example:
        >>> template = get_prompt_template('SENTIMENT_ANALYSIS')
        >>> prompt = template.substitute(text="这是一条利好消息...")
    """
    if not hasattr(PromptTemplates, template_name):
        raise AttributeError(f"Template '{template_name}' not found in PromptTemplates")

    return getattr(PromptTemplates, template_name)


def format_prompt(template_name: str, **kwargs) -> str:
    """
    Format a prompt template with provided variables.

    Args:
        template_name: Name of the template
        **kwargs: Variables to substitute in the template

    Returns:
        str: Formatted prompt string

    Example:
        >>> prompt = format_prompt('SENTIMENT_ANALYSIS', text="利好消息...")
    """
    template = get_prompt_template(template_name)
    return template.substitute(**kwargs)


def list_templates() -> List[str]:
    """
    List all available prompt templates.

    Returns:
        List[str]: List of template names

    Example:
        >>> templates = list_templates()
        >>> print(templates)
        ['SENTIMENT_ANALYSIS', 'STOCK_SENTIMENT', ...]
    """
    return [
        attr for attr in dir(PromptTemplates)
        if attr.isupper() and isinstance(getattr(PromptTemplates, attr), Template)
    ]


def create_custom_template(template_str: str) -> Template:
    """
    Create a custom prompt template.

    Args:
        template_str: Template string with $variable placeholders

    Returns:
        Template: Custom template object

    Example:
        >>> template = create_custom_template("分析 $stock 的 $indicator 指标")
        >>> prompt = template.substitute(stock="贵州茅台", indicator="RSI")
    """
    return Template(template_str)


# ============================================
# 便捷函数
# ============================================

def analyze_sentiment_prompt(text: str, stock_name: Optional[str] = None) -> str:
    """
    Generate a sentiment analysis prompt.

    Args:
        text: Text to analyze
        stock_name: Stock name for stock-specific analysis (optional)

    Returns:
        str: Formatted prompt
    """
    if stock_name:
        # 需要同时提供 stock_name 和 stock_code
        return PromptTemplates.STOCK_SENTIMENT.substitute(
            stock_name=stock_name,
            stock_code="",
            text=text
        )
    return PromptTemplates.SENTIMENT_ANALYSIS.substitute(text=text)


def summarize_prompt(text: str, summary_type: str = "general") -> str:
    """
    Generate a summarization prompt.

    Args:
        text: Text to summarize
        summary_type: Type of summary ('general' or 'earnings')

    Returns:
        str: Formatted prompt
    """
    if summary_type == "earnings":
        return PromptTemplates.EARNINGS_SUMMARY.substitute(text=text)
    return PromptTemplates.FINANCIAL_SUMMARY.substitute(text=text)


def extract_entities_prompt(text: str) -> str:
    """
    Generate an entity extraction prompt.

    Args:
        text: Text to extract entities from

    Returns:
        str: Formatted prompt
    """
    return PromptTemplates.ENTITY_EXTRACTION.substitute(text=text)
