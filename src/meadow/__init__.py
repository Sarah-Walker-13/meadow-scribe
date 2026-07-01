"""MeadowScribe — Markdown-to-PDF converter with frontmatter extraction."""

__version__ = "0.6.0"
__author__ = "Meadow Scribe Maintainers"
__all__ = ["MarkdownConverter", "FrontmatterExtractor", "PDFTemplate", "ConversionResult"]

from .converter import MarkdownConverter, PDFTemplate, ConversionResult
from .extractor import FrontmatterExtractor
