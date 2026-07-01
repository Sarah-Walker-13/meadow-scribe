"""
Rule-based domain extractor for Chinese text documents.

Assigns domain labels (tech, literature, conversation, academic, legal,
social-media) using keyword frequency analysis and structural heuristics.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .converter import ConversionResult


class FrontmatterField(Enum):
    TECHNOLOGY = "tech"
    LITERATURE = "literature"
    CONVERSATION = "conversation"
    ACADEMIC = "academic"
    LEGAL = "legal"
    SOCIAL_MEDIA = "social-media"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    label: FrontmatterField
    confidence: float
    scores: Dict[FrontmatterField, float] = field(default_factory=dict)
    top_keywords: List[str] = field(default_factory=list)


class FrontmatterExtractor:
    """Classifies Chinese text into domain categories.

    Uses weighted keyword sets, structural pattern matching, and
    density heuristics. No ML dependencies — pure rule-based.
    """

    # Weighted domain keyword sets (Chinese)
    TECH_KEYWORDS: Dict[str, float] = {
        "代码": 3.0, "函数": 3.0, "接口": 3.0, "算法": 3.5, "数据": 2.5,
        "服务器": 3.0, "编译": 3.0, "部署": 3.0, "测试": 2.5, "调试": 3.0,
        "架构": 3.5, "模块": 2.5, "依赖": 2.5, "性能": 3.0, "优化": 2.5,
        "缓存": 3.0, "数据库": 3.5, "查询": 2.5, "索引": 3.0, "事务": 3.0,
        "异步": 3.0, "并发": 3.5, "线程": 3.0, "进程": 3.0, "容器": 2.5,
        "容器化": 3.5, "微服务": 3.5, "分布式": 3.5, "集群": 3.0,
        "API": 3.5, "SDK": 3.0, "HTTP": 2.5, "JSON": 2.5, "REST": 3.0,
        "前端": 2.5, "后端": 2.5, "全栈": 3.0, "框架": 2.5, "库": 1.5,
        "Python": 2.0, "Java": 2.0, "Go": 2.0, "Rust": 2.5, "TypeScript": 2.0,
        "bug": 2.0, "fix": 2.0, "feature": 2.0, "commit": 2.0, "PR": 2.0,
        "README": 2.0, "TODO": 1.5, "FIXME": 2.0, "HACK": 2.0,
    }

    LITERATURE_KEYWORDS: Dict[str, float] = {
        "月光": 4.0, "星空": 3.5, "落叶": 3.5, "春风": 3.5, "秋风": 3.0,
        "黄昏": 3.5, "黎明": 3.5, "暮色": 4.0, "晨光": 3.5, "晚霞": 4.0,
        "孤独": 3.5, "思念": 3.5, "回忆": 3.0, "梦": 2.0, "远方": 3.0,
        "故乡": 3.5, "母亲": 3.0, "父亲": 3.0, "童年": 3.5, "青春": 3.0,
        "生命": 3.0, "命运": 3.5, "时光": 3.0, "岁月": 3.0, "流年": 4.0,
        "繁华": 3.5, "凋零": 4.0, "绽放": 3.5, "芬芳": 4.0, "微风": 3.5,
        "描写": 3.0, "抒情": 3.5, "修辞": 3.5, "意境": 4.0, "意象": 3.5,
        "一章": 2.0, "序章": 2.5, "尾声": 2.5, "幕": 2.0,
    }

    ACADEMIC_KEYWORDS: Dict[str, float] = {
        "研究": 3.0, "分析": 2.5, "实验": 3.5, "证明": 3.0, "假设": 3.5,
        "结论": 3.0, "引言": 3.5, "摘要": 3.5, "参考文献": 4.0, "引用": 2.5,
        "理论": 3.5, "实践": 2.5, "方法论": 4.0, "模型": 2.5, "框架": 2.0,
        "统计": 3.0, "显著": 3.0, "样本": 3.5, "变量": 3.5, "系数": 3.5,
        "相关": 2.5, "显著差异": 4.0, "结果表明": 4.0, "本文": 3.0,
        "笔者": 3.0, "学界": 3.5, "学术": 3.5, "学科": 3.0, "领域": 2.5,
        "贡献": 2.5, "创新": 2.5, "突破": 2.5,
    }

    CONVERSATION_KEYWORDS: Dict[str, float] = {
        "你好": 4.0, "谢谢": 4.0, "请问": 3.5, "麻烦": 3.0, "不好意思": 3.5,
        "哈哈": 3.5, "嗯嗯": 3.5, "好的": 3.0, "没问题": 3.0, "可以吗": 3.5,
        "怎么样": 3.0, "对不": 3.0, "真的": 2.0, "感觉": 2.0, "觉得": 2.0,
        "应该": 2.0, "可能": 2.0, "或者": 2.0, "不过": 2.0, "但是": 2.0,
        "其实": 2.5, "反正": 2.5, "毕竟": 2.5, "总之": 2.5, "然后": 2.0,
        "那个": 2.5, "这个": 2.0, "什么": 1.5, "怎么": 1.5, "哪里": 2.0,
    }

    SOCIAL_MEDIA_KEYWORDS: Dict[str, float] = {
        "转发": 3.5, "评论": 3.0, "点赞": 3.5, "关注": 2.5, "粉丝": 3.0,
        "分享": 2.5, "收藏": 2.5, "举报": 3.0, "话题": 3.0, "热搜": 3.5,
        "视频": 2.0, "直播": 3.0, "博主": 3.5, "网友": 3.0, "评论区": 4.0,
        "朋友": 2.0, "小红书": 4.0, "微博": 4.0, "微信": 3.0, "抖音": 3.5,
        "UP主": 4.0, "弹幕": 4.0, "三连": 4.5, "充电": 3.5, "投币": 4.0,
        "家人们": 4.5, "姐妹们": 4.5, "兄弟们": 4.0, "亲们": 4.0,
    }

    LEGAL_KEYWORDS: Dict[str, float] = {
        "合同": 4.0, "甲方": 4.5, "乙方": 4.5, "条款": 3.5, "协议": 3.5,
        "法律": 3.5, "依法": 3.5, "承担": 2.5, "责任": 2.5, "义务": 3.0,
        "权利": 3.0, "授权": 3.5, "许可": 3.0, "违约": 4.0, "赔偿": 4.0,
        "管辖": 4.0, "仲裁": 4.0, "法院": 4.0, "判决": 4.0, "法律效力": 4.5,
        "知识产权": 4.5, "保密": 3.5, "隐私": 3.5, "合规": 3.5,
    }

    # Structural patterns
    STRUCTURAL_PATTERNS: Dict[FrontmatterField, List[str]] = {
        FrontmatterField.TECHNOLOGY: [
            r"(?m)^\s*(def|class|import|from|if __name__)",
            r"(?m)^\s*(func|package|struct|interface)\s",
            r"(?m)^\s*(const|let|var)\s+\w+\s*=",
            r"(?m)^[#{}/\s]*(TODO|FIXME|HACK|XXX|NOTE)",
            r"```[\s\S]*?```",
            r"https?://github\.com/",
        ],
        FrontmatterField.ACADEMIC: [
            r"\[\d+\]",
            r"(?m)^\s*abstract\b",
            r"et al\.?",
            r"doi:",
            r"arXiv:",
        ],
        FrontmatterField.SOCIAL_MEDIA: [
            r"#[一-鿿\w]+",
            r"@\S+",
            r"//@",
        ],
        FrontmatterField.CONVERSATION: [
            r"[。！？~～]{2,}",
            r"[哈嘿嘻呵诶哎嗨]+",
            r"\b(hhh+|www+|233|666)\b",
        ],
    }

    def __init__(self):
        self._all_keywords = self._build_keyword_set()

    def extract(self, doc: ConversionResult) -> ClassificationResult:
        """Classify a document into the best-matching domain."""
        scores: Dict[FrontmatterField, float] = {}
        text = doc.content

        # Keyword scoring
        keyword_sets: Dict[FrontmatterField, Dict[str, float]] = {
            FrontmatterField.TECHNOLOGY: self.TECH_KEYWORDS,
            FrontmatterField.LITERATURE: self.LITERATURE_KEYWORDS,
            FrontmatterField.ACADEMIC: self.ACADEMIC_KEYWORDS,
            FrontmatterField.CONVERSATION: self.CONVERSATION_KEYWORDS,
            FrontmatterField.SOCIAL_MEDIA: self.SOCIAL_MEDIA_KEYWORDS,
            FrontmatterField.LEGAL: self.LEGAL_KEYWORDS,
        }
        matched: Dict[FrontmatterField, List[Tuple[str, float]]] = {}
        for label, keywords in keyword_sets.items():
            score = 0.0
            matched[label] = []
            for kw, weight in keywords.items():
                count = text.count(kw)
                if count > 0:
                    s = weight * count
                    # Bonus for keyword density
                    if len(text) > 0 and count / len(text) > 0.001:
                        s *= 1.5
                    score += s
                    matched[label].append((kw, s))
            # Normalize by document length
            score = score / max(len(text) / 100, 1)
            scores[label] = score

        # Structural bonus
        for label, patterns in self.STRUCTURAL_PATTERNS.items():
            for pat in patterns:
                matches = len(re.findall(pat, text))
                if matches > 0:
                    scores[label] = scores.get(label, 0) + matches * 2.0

        # File extension hints
        ext = doc.metadata.get("file_extension", "")
        ext_hints = {
            ".py": FrontmatterField.TECHNOLOGY, ".js": FrontmatterField.TECHNOLOGY,
            ".ts": FrontmatterField.TECHNOLOGY, ".go": FrontmatterField.TECHNOLOGY,
            ".rs": FrontmatterField.TECHNOLOGY, ".java": FrontmatterField.TECHNOLOGY,
            ".cpp": FrontmatterField.TECHNOLOGY, ".c": FrontmatterField.TECHNOLOGY,
            ".h": FrontmatterField.TECHNOLOGY,
            ".md": FrontmatterField.TECHNOLOGY,  # Most .md files are tech docs
        }
        if ext in ext_hints:
            scores[ext_hints[ext]] = scores.get(ext_hints[ext], 0) + 3.0

        # Determine best label
        best_label = FrontmatterField.UNKNOWN
        best_score = 0.0
        for label, score in scores.items():
            if score > best_score:
                best_score = score
                best_label = label

        # Confidence: ratio of best to second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        confidence = 0.0
        if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
            second = sorted_scores[1]
            margin = (sorted_scores[0] - second) / sorted_scores[0]
            confidence = min(1.0, sorted_scores[0] / max(second, 0.01) / 5)
        elif sorted_scores and sorted_scores[0] > 0:
            confidence = 0.5

        if best_label == FrontmatterField.UNKNOWN:
            confidence = 0.35
            # Fallback: check if general Chinese content
            if doc.chinese_ratio > 0.3:
                best_label = FrontmatterField.GENERAL
                confidence = 0.4

        top_kw = sorted(matched.get(best_label, []), key=lambda x: -x[1])[:5]

        return ClassificationResult(
            label=best_label,
            confidence=round(confidence, 3),
            scores={k: round(v, 2) for k, v in scores.items()},
            top_keywords=[kw for kw, _ in top_kw],
        )

    def batch_classify(
        self, documents: List[ConversionResult]
    ) -> List[Tuple[ConversionResult, ClassificationResult]]:
        return [(doc, self.extract(doc)) for doc in documents]

    def _build_keyword_set(self) -> Set[str]:
        all_kw: Set[str] = set()
        for mapping in [
            self.TECH_KEYWORDS, self.LITERATURE_KEYWORDS,
            self.ACADEMIC_KEYWORDS, self.CONVERSATION_KEYWORDS,
            self.SOCIAL_MEDIA_KEYWORDS, self.LEGAL_KEYWORDS,
        ]:
            all_kw.update(mapping.keys())
        return all_kw

    @staticmethod
    def distribution(
        results: List[ClassificationResult],
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in results:
            key = r.label.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
