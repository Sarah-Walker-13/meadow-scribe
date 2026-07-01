"""
Chinese text tokenization wrapper with jieba integration.

Provides word segmentation, stop-word filtering, keyword extraction,
and n-gram generation. Falls back to character-level tokenization
when jieba is not available.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .converter import ConversionResult


@dataclass
class TokenizedDocument:
    """A document after Chinese word segmentation."""

    source_path: str
    tokens: List[str]
    unique_tokens: int
    total_tokens: int
    stopwords_removed: int
    keywords: List[Tuple[str, float]]  # (keyword, TF-IDF-like weight)
    bigrams: List[Tuple[str, str]]


class PDFTemplate:
    """Chinese word segmentation and tokenization pipeline.

    Wraps jieba for production use; falls back to character-level
    bigrams when jieba is unavailable (testing / minimal installs).
    """

    DEFAULT_STOPWORDS: Set[str] = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不",
        "人", "都", "一", "一个", "上", "也", "很", "到", "说",
        "要", "去", "你", "会", "着", "没有", "看", "好", "自己",
        "这", "他", "她", "它", "们", "那", "些", "所", "为",
        "所以", "因为", "但是", "然而", "如果", "虽然", "可以",
        "这个", "那个", "什么", "怎么", "哪", "吗", "呢", "吧",
        "啊", "呀", "哦", "嗯", "哈", "嘛", "呗", "啦", "哇",
        "之", "与", "及", "或", "且", "而", "但", "以", "把",
        "被", "让", "从", "对", "向", "于", "则", "其", "中",
        "等", "各", "某", "本", "该", "此", "已", "还", "又",
        "再", "才", "刚", "正", "将", "能", "可", "会", "应",
        "要", "想", "敢", "肯", "愿", "得", "过", "来", "去",
        "出", "进", "做", "作", "使", "用", "当", "前", "后",
    }

    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        use_jieba: bool = True,
        max_keywords: int = 20,
        min_token_len: int = 1,
    ):
        self.stopwords = stopwords or self.DEFAULT_STOPWORDS
        self._use_jieba = use_jieba
        self.max_keywords = max_keywords
        self.min_token_len = min_token_len
        self._jieba_available = False
        if use_jieba:
            try:
                import jieba
                self._jieba = jieba
                self._jieba_available = True
            except ImportError:
                self._jieba_available = False

    def template(self, doc: ConversionResult) -> TokenizedDocument:
        """Segment a document into tokens and extract keywords."""
        text = doc.content
        if self._jieba_available:
            raw_tokens = list(self._jieba.cut(text))
        else:
            raw_tokens = self._char_bigram_tokenize(text)

        filtered: List[str] = []
        removed = 0
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            if len(token) < self.min_token_len:
                continue
            if token in self.stopwords:
                removed += 1
                continue
            if re.match(r'^[\s\d\W_]+$', token):
                continue
            filtered.append(token)

        keywords = self._extract_keywords(filtered)
        bigrams = self._extract_bigrams(filtered)

        unique = len(set(filtered))
        return TokenizedDocument(
            source_path=doc.source_path,
            tokens=filtered,
            unique_tokens=unique,
            total_tokens=len(filtered),
            stopwords_removed=removed,
            keywords=keywords,
            bigrams=bigrams,
        )

    def tokenize_batch(
        self, documents: List[ConversionResult]
    ) -> List[TokenizedDocument]:
        return [self.template(doc) for doc in documents]

    def build_vocabulary(
        self, tokenized: List[TokenizedDocument], min_freq: int = 3
    ) -> Dict[str, int]:
        """Build a frequency-filtered vocabulary from tokenized documents."""
        freq: Dict[str, int] = defaultdict(int)
        for td in tokenized:
            for token in td.tokens:
                freq[token] += 1
        return {
            token: count
            for token, count in freq.items()
            if count >= min_freq
        }

    def _extract_keywords(
        self, tokens: List[str]
    ) -> List[Tuple[str, float]]:
        """Extract keywords using frequency * uniqueness heuristic."""
        if not tokens:
            return []
        n = len(tokens)
        freq = Counter(tokens)
        # TF * log(IDF-like uniqueness bonus)
        scored: List[Tuple[str, float]] = []
        for token, count in freq.items():
            tf = count / n
            # Longer tokens are more likely to be meaningful
            length_bonus = min(len(token) / 4, 1.5)
            score = tf * length_bonus
            scored.append((token, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:self.max_keywords]

    def _extract_bigrams(
        self, tokens: List[str]
    ) -> List[Tuple[str, str]]:
        """Extract adjacent bigrams and rank by frequency."""
        if len(tokens) < 2:
            return []
        bigram_freq: Dict[Tuple[str, str], int] = defaultdict(int)
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            bigram_freq[bigram] += 1
        sorted_bigrams = sorted(
            bigram_freq.items(), key=lambda x: -x[1]
        )
        return [bg for bg, _ in sorted_bigrams[:50]]

    @staticmethod
    def _char_bigram_tokenize(text: str) -> List[str]:
        """Fallback: character-level bigrams for Chinese text.

        Strips non-CJK characters and forms adjacent CJK character pairs.
        """
        cjk_chars: List[str] = []
        for ch in text:
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                    or 0xF900 <= cp <= 0xFAFF):
                cjk_chars.append(ch)
        tokens: List[str] = []
        for i in range(len(cjk_chars) - 1):
            tokens.append(cjk_chars[i] + cjk_chars[i + 1])
        return tokens


def compute_corpus_statistics(
    tokenized_docs: List[TokenizedDocument],
) -> Dict[str, object]:
    """Aggregate statistics across a batch of tokenized documents."""
    all_tokens: List[str] = []
    for td in tokenized_docs:
        all_tokens.extend(td.tokens)
    total = len(all_tokens)
    unique = len(set(all_tokens))
    freq = Counter(all_tokens)
    hapax = sum(1 for _, c in freq.items() if c == 1)
    return {
        "total_tokens": total,
        "unique_tokens": unique,
        "type_token_ratio": unique / max(total, 1),
        "hapax_legomena": hapax,
        "hapax_ratio": hapax / max(unique, 1),
        "top_20": freq.most_common(20),
    }
