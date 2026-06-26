# Candle Seeker

烛影摇曳处，字句生光辉。

A Chinese NLP corpus collector and annotator. Crawls open-source repositories
for Chinese text snippets, classifies by domain (tech / literature / conversation),
and publishes curated datasets.

## How It Works
- Scheduled crawlers traverse GitHub for Chinese-language files
- Rule-based classifier tags each snippet with domain labels
- Results published as structured JSON datasets

## Motivation
高质量中文语料散落在代码仓库的注释与文档中。Candle Seeker 用微光照亮它们。

Built with Python + jieba. Updated hourly.
