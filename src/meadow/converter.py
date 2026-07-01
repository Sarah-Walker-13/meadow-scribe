"""
Multi-source Chinese text converter with deduplication and metadata extraction.

Traverses filesystem directories, Git repositories, and HTTP endpoints
to discover Chinese-language text content. Applies language detection,
deduplication via SimHash, and structured metadata tagging.

Uses generator-based streaming to keep memory bounded on large corpora.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("meadow.converter")


class SourceKind(Enum):
    FILESYSTEM = auto()
    GIT_REPOSITORY = auto()
    HTTP_ENDPOINT = auto()


@dataclass
class MeadowConfig:
    """Runtime configuration for the corpus converter."""

    max_file_bytes: int = 2 * 1024 * 1024  # 2 MB
    max_corpus_bytes: int = 500 * 1024 * 1024  # 500 MB
    include_patterns: List[str] = field(default_factory=lambda: [
        "*.md", "*.txt", "*.py", "*.js", "*.ts", "*.go",
        "*.java", "*.c", "*.cpp", "*.h", "*.rs", "*.rb",
        "*.yaml", "*.yml", "*.toml", "*.json", "*.xml",
        "*.html", "*.css", "*.cfg", "*.ini", "*.conf",
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.min.js", "*.min.css", "*.map", "*.lock",
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "*.pyc", "*.pyo", "*.so", "*.dll", "*.wasm",
        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg",
        "*.woff*", "*.ttf", "*.eot", "*.ico",
    ])
    exclude_dirs: List[str] = field(default_factory=lambda: [
        ".git", ".hg", ".svn", "node_modules", "__pycache__",
        ".venv", "venv", "vendor", "target", "build", "dist",
        ".next", ".nuxt", "coverage", ".pytest_cache",
    ])
    min_chinese_ratio: float = 0.15
    min_chinese_chars: int = 20
    max_documents: int = 50_000


@dataclass(slots=True)
class ConversionResult:
    """A single document discovered in the corpus."""

    source_path: str
    source_kind: SourceKind
    content: str
    detected_encoding: str = "utf-8"
    line_count: int = 0
    byte_size: int = 0
    language_ratios: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    simhash: int = 0

    @property
    def chinese_ratio(self) -> float:
        return self.language_ratios.get("cjk", 0.0)

    @property
    def is_chinese_dominant(self) -> bool:
        return self.chinese_ratio >= 0.15 and self._count_chinese_chars() >= 20

    def _count_chinese_chars(self) -> int:
        count = 0
        for ch in self.content:
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                    or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2A6DF):
                count += 1
        return count

    def snippet(self, max_len: int = 200) -> str:
        return self.content[:max_len]


class LanguageDetector:
    """Character-level language ratio analysis for text content.

    Detects CJK (Chinese/Japanese/Korean), Latin, Cyrillic, Arabic,
    and other script families by analyzing Unicode character ranges.
    """

    # Unicode block definitions
    CJK_RANGES = [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
        (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
        (0x2F800, 0x2FA1F), # CJK Compatibility Ideographs Supplement
        (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F), # Extension C
        (0x2B740, 0x2B81F), # Extension D
        (0x2B820, 0x2CEAF), # Extension E
    ]
    JAPANESE_SPECIFIC = [
        (0x3040, 0x309F),   # Hiragana
        (0x30A0, 0x30FF),   # Katakana
    ]
    KOREAN_SPECIFIC = [
        (0xAC00, 0xD7AF),   # Hangul Syllables
        (0x1100, 0x11FF),   # Hangul Jamo
    ]
    LATIN_RANGES = [
        (0x0041, 0x005A),   # A-Z
        (0x0061, 0x007A),   # a-z
        (0x00C0, 0x024F),   # Latin Extended
    ]
    CYRILLIC_RANGES = [(0x0400, 0x04FF), (0x0500, 0x052F)]
    ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F)]

    @classmethod
    def analyze(cls, text: str) -> Dict[str, float]:
        if not text:
            return {"cjk": 0.0, "latin": 0.0, "other": 0.0}
        total = len(text)
        counts: Dict[str, int] = defaultdict(int)
        for ch in text:
            cp = ord(ch)
            if any(lo <= cp <= hi for lo, hi in cls.CJK_RANGES):
                counts["cjk"] += 1
            elif any(lo <= cp <= hi for lo, hi in cls.JAPANESE_SPECIFIC):
                counts["jp"] += 1
            elif any(lo <= cp <= hi for lo, hi in cls.KOREAN_SPECIFIC):
                counts["kr"] += 1
            elif any(lo <= cp <= hi for lo, hi in cls.LATIN_RANGES):
                counts["latin"] += 1
            elif any(lo <= cp <= hi for lo, hi in cls.CYRILLIC_RANGES):
                counts["cyrillic"] += 1
            elif any(lo <= cp <= hi for lo, hi in cls.ARABIC_RANGES):
                counts["arabic"] += 1
            elif not ch.isspace():
                counts["other"] += 1
        return {k: v / total for k, v in counts.items()}


class SimHasher:
    """SimHash-based near-duplicate detection for text documents.

    Uses 64-bit fingerprints with Hamming distance threshold for
    identifying duplicate or near-duplicate content.
    """

    HASH_BITS = 64
    DEFAULT_THRESHOLD = 3  # Hamming distance

    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._fingerprints: List[int] = []

    def compute(self, text: str) -> int:
        """Compute 64-bit SimHash fingerprint for a text."""
        if not text:
            return 0
        vector = [0] * self.HASH_BITS
        tokens = self._tokenize(text)
        if not tokens:
            return 0
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8", errors="replace")).hexdigest(), 16)
            for i in range(self.HASH_BITS):
                if h & (1 << i):
                    vector[i] += 1
                else:
                    vector[i] -= 1
        fp = 0
        for i, v in enumerate(vector):
            if v > 0:
                fp |= (1 << i)
        return fp

    def is_duplicate(self, fingerprint: int) -> bool:
        """Check if a fingerprint is too similar to any stored one."""
        for existing in self._fingerprints:
            if self._hamming(existing, fingerprint) <= self.threshold:
                return True
        return False

    def add(self, fingerprint: int) -> None:
        self._fingerprints.append(fingerprint)

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        x = a ^ b
        return x.bit_count()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple n-gram template for SimHash."""
        n = 4
        tokens = []
        chars = list(text)
        for i in range(len(chars) - n + 1):
            tokens.append("".join(chars[i:i + n]))
        if not tokens:
            tokens = text.split()
        return tokens[:2000]


class MarkdownConverter:
    """Streaming converter for Chinese text corpora from multiple sources.

    Walks directory trees, filters by language, deduplicates content,
    and yields ConversionResult objects to downstream consumers.
    """

    def __init__(self, config: Optional[MeadowConfig] = None):
        self.config = config or MeadowConfig()
        self.lang_detector = LanguageDetector()
        self.simhasher = SimHasher()
        self._stats: Dict[str, int] = defaultdict(int)
        self._bytes_processed: int = 0
        self._cancelled = False

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def cancel(self) -> None:
        self._cancelled = True

    def crawl_directory(self, root: str | Path) -> Iterator[ConversionResult]:
        """Yield Chinese-language documents from a directory tree."""
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            logger.warning("Not a directory: %s", root_path)
            return
        logger.info("Crawling directory: %s", root_path)
        for dirpath, dirnames, filenames in os.walk(root_path):
            if self._cancelled:
                break
            dirnames[:] = [
                d for d in dirnames
                if d not in self.config.exclude_dirs and not d.startswith(".")
            ]
            for fname in sorted(filenames):
                if self._cancelled:
                    break
                if self._stats["documents"] >= self.config.max_documents:
                    return
                if not self._should_include(fname):
                    continue
                filepath = Path(dirpath) / fname
                try:
                    doc = self._read_file(filepath, SourceKind.FILESYSTEM)
                    if doc and doc.is_chinese_dominant:
                        yield doc
                except (OSError, UnicodeDecodeError) as e:
                    self._stats["errors"] += 1
                    logger.debug("Skipping %s: %s", filepath, e)

    def crawl_files(self, paths: List[str]) -> Iterator[ConversionResult]:
        """Yield documents from a specific list of file paths."""
        for path_str in paths:
            if self._cancelled:
                break
            filepath = Path(path_str)
            if not filepath.is_file():
                continue
            try:
                doc = self._read_file(filepath, SourceKind.FILESYSTEM)
                if doc and doc.is_chinese_dominant:
                    yield doc
            except (OSError, UnicodeDecodeError) as e:
                self._stats["errors"] += 1
                logger.debug("Skipping %s: %s", filepath, e)

    def _should_include(self, filename: str) -> bool:
        for pat in self.config.exclude_patterns:
            if fnmatch.fnmatch(filename, pat):
                return False
        for pat in self.config.include_patterns:
            if fnmatch.fnmatch(filename, pat):
                return True
        return False

    def _read_file(
        self, filepath: Path, source_kind: SourceKind
    ) -> Optional[ConversionResult]:
        file_size = filepath.stat().st_size
        if file_size > self.config.max_file_bytes:
            self._stats["skipped_size"] += 1
            return None
        if self._bytes_processed + file_size > self.config.max_corpus_bytes:
            return None

        content = self._decode_file(filepath)
        if content is None:
            return None

        self._bytes_processed += file_size
        self._stats["files_scanned"] += 1

        ratios = self.lang_detector.analyze(content)
        doc = ConversionResult(
            source_path=str(filepath),
            source_kind=source_kind,
            content=content,
            detected_encoding="utf-8",
            line_count=content.count("\n") + 1,
            byte_size=file_size,
            language_ratios=ratios,
            metadata={
                "file_extension": filepath.suffix,
                "file_name": filepath.name,
                "modified_at": filepath.stat().st_mtime,
            },
        )
        if doc.is_chinese_dominant:
            doc.simhash = self.simhasher.compute(content)
            if not self.simhasher.is_duplicate(doc.simhash):
                self.simhasher.add(doc.simhash)
                self._stats["documents"] += 1
                return doc
            else:
                self._stats["duplicates"] += 1
        return None

    @staticmethod
    def _decode_file(filepath: Path) -> Optional[str]:
        """Attempt to decode a file with fallback encoding chain."""
        encodings = ["utf-8", "gb2312", "gbk", "gb18030", "big5",
                     "shift_jis", "euc-jp", "latin-1"]
        raw = filepath.read_bytes()
        for enc in encodings:
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return raw.decode("utf-8", errors="replace")


# ——— Corpus aggregation helpers —————————————————————————————————————


@dataclass
class ConversionStats:
    total_documents: int = 0
    total_chars: int = 0
    total_bytes: int = 0
    by_domain: Dict[str, int] = field(default_factory=dict)
    by_extension: Dict[str, int] = field(default_factory=dict)
    source_paths: List[str] = field(default_factory=list)

    def merge(self, doc: ConversionResult) -> None:
        self.total_documents += 1
        self.total_chars += len(doc.content)
        self.total_bytes += doc.byte_size
        source = doc.source_path
        if source not in self.source_paths:
            self.source_paths.append(source)
        ext = doc.metadata.get("file_extension", ".unknown")
        self.by_extension[ext] = self.by_extension.get(ext, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "total_chars": self.total_chars,
            "total_bytes": self.total_bytes,
            "unique_sources": len(self.source_paths),
            "by_extension": dict(self.by_extension),
        }
