"""Tests for MeadowScribe corpus converter and extractor."""

from meadow.converter import (
    MarkdownConverter,
    ConversionResult,
    LanguageDetector,
    MeadowConfig,
    SimHasher,
    SourceKind,
)
from meadow.extractor import FrontmatterExtractor, FrontmatterField, ClassificationResult
from meadow.template import PDFTemplate, TokenizedDocument, compute_corpus_statistics


class TestLanguageDetector:
    def test_pure_chinese(self):
        text = "这是一段纯中文文本用于测试语言检测功能"
        ratios = LanguageDetector.analyze(text)
        assert ratios["cjk"] > 0.8

    def test_pure_english(self):
        text = "This is pure English text for testing"
        ratios = LanguageDetector.analyze(text)
        assert ratios["latin"] > 0.8

    def test_mixed_content(self):
        text = "Chinese中文 and English mixed 混合内容"
        ratios = LanguageDetector.analyze(text)
        assert ratios["cjk"] > 0.2
        assert ratios["latin"] > 0.2

    def test_empty_text(self):
        ratios = LanguageDetector.analyze("")
        assert ratios["cjk"] == 0.0


class TestSimHasher:
    def test_compute(self):
        sh = SimHasher()
        fp1 = sh.compute("这是第一段文本")
        fp2 = sh.compute("这是一段不同的文本内容")
        assert fp1 != 0
        assert fp2 != 0
        assert fp1 != fp2

    def test_duplicate_detection(self):
        sh = SimHasher()
        fp1 = sh.compute("重复的文本内容用于测试去重功能")
        sh.add(fp1)
        fp2 = sh.compute("重复的文本内容用于测试去重功能")
        assert sh.is_duplicate(fp2)

    def test_near_duplicate(self):
        sh = SimHasher(threshold=3)
        fp1 = sh.compute("这是一段较长的文本用于测试近似去重功能" * 10)
        sh.add(fp1)
        fp2 = sh.compute("这是一段较长的文本用于测试近似去重功能" * 10)
        assert sh.is_duplicate(fp2)


class TestCorpusDocument:
    def test_chinese_detection(self):
        doc = ConversionResult(
            source_path="/test/file.md",
            source_kind=SourceKind.FILESYSTEM,
            content="这是一段包含足够数量中文字符的文本文档内容" * 3,
        )
        assert doc.is_chinese_dominant

    def test_not_chinese(self):
        doc = ConversionResult(
            source_path="/test/readme.md",
            source_kind=SourceKind.FILESYSTEM,
            content="This is just English text without any Chinese characters at all.",
        )
        assert not doc.is_chinese_dominant


class TestFrontmatterExtractor:
    def test_tech_classification(self):
        extractor = FrontmatterExtractor()
        doc = ConversionResult(
            source_path="/src/main.py",
            source_kind=SourceKind.FILESYSTEM,
            content="这个函数实现了数据库查询的优化功能代码编译部署",
            metadata={"file_extension": ".py"},
        )
        result = extractor.extract(doc)
        assert result.label == FrontmatterField.TECHNOLOGY
        assert result.confidence > 0

    def test_literature_classification(self):
        extractor = FrontmatterExtractor()
        doc = ConversionResult(
            source_path="/poem.txt",
            source_kind=SourceKind.FILESYSTEM,
            content="月光如水洒在窗前，微风轻拂落叶，意境深远，描写细腻的抒情散文",
        )
        result = extractor.extract(doc)
        assert result.label in (FrontmatterField.LITERATURE, FrontmatterField.GENERAL)

    def test_classification_result_has_scores(self):
        extractor = FrontmatterExtractor()
        doc = ConversionResult(
            source_path="/test.md",
            source_kind=SourceKind.FILESYSTEM,
            content="这是一段技术文档关于API和数据库设计架构",
            metadata={"file_extension": ".md"},
        )
        result = extractor.extract(doc)
        assert len(result.scores) > 0
        assert result.confidence > 0

    def test_distribution(self):
        extractor = FrontmatterExtractor()
        results = [
            ClassificationResult(FrontmatterField.TECHNOLOGY, 0.8),
            ClassificationResult(FrontmatterField.TECHNOLOGY, 0.7),
            ClassificationResult(FrontmatterField.LITERATURE, 0.9),
        ]
        dist = extractor.distribution(results)
        assert dist["tech"] == 2
        assert dist["literature"] == 1


class TestPDFTemplate:
    def test_bigram_fallback(self):
        template = PDFTemplate(use_jieba=False)
        doc = ConversionResult(
            source_path="/test.txt",
            source_kind=SourceKind.FILESYSTEM,
            content="中文文本分词测试内容",
        )
        td = template.template(doc)
        assert len(td.tokens) > 0
        assert isinstance(td.keywords, list)

    def test_stopword_filtering(self):
        template = PDFTemplate(use_jieba=False)
        doc = ConversionResult(
            source_path="/test.txt",
            source_kind=SourceKind.FILESYSTEM,
            content="这是一个测试的文档内容",
        )
        td = template.template(doc)
        assert td.stopwords_removed >= 0

    def test_compute_statistics(self):
        template = PDFTemplate(use_jieba=False)
        docs = [
            ConversionResult(
                source_path=f"/test{i}.txt",
                source_kind=SourceKind.FILESYSTEM,
                content="中文文本分词测试内容数据处理分析",
            )
            for i in range(3)
        ]
        tokenized = template.tokenize_batch(docs)
        stats = compute_corpus_statistics(tokenized)
        assert stats["total_tokens"] > 0
        assert stats["unique_tokens"] > 0
        assert 0 < stats["type_token_ratio"] <= 1.0
