"""Language of user-visible explanations; schema and source text stay stable."""

CHINESE_EXPLANATIONS = """
用户面向的解释必须使用简体中文：包括识别依据、reason、recognition_evidence、
knowledge_evidence、legend_evidence、source_evidence、visible_strokes、描述、
不确定性说明和排除原因。说明具体可见的笔画、文字或位置，不要只说“符合标准”。
不要翻译 JSON 字段名、枚举值、引用编号，也不要改写原图位号、管线编号或逐字引用。
未确定的类型应保留不确定，不得为满足中文要求虚构证据。
"""
