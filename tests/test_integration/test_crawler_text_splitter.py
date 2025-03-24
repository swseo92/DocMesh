from collections import defaultdict
from docmesh.text_splitter import DocumentChunkPipeline, default_tokenizer


def test_crawler_pipeline_integration(monkeypatch):
    # Import WebCrawler from crawler.py and DocumentChunkPipeline from our module.
    from docmesh.crawler import WebCrawler

    content1 = "This is the first paragraph under heading 1." * 100
    content2 = "This is a paragraph under heading 2." * 100
    content3 = "This is another paragraph under heading 2." * 100

    # Fake HTML and FakePageFetcher as before
    FAKE_HTML = f"""
    <html>
    <head><title>Test Page</title></head>
    <body>
    <h1>Heading 1</h1>
    <p>{content1}</p>
    <h2>Heading 2</h2>
    <p>{content2}</p>
    <p>{content3}</p>
    </body>
    </html>
    """

    FAKE_CONTENT_TYPE = "text/html"

    class FakePageFetcher:
        def fetch(self, url: str):
            return (FAKE_HTML, FAKE_CONTENT_TYPE)

    # Create a WebCrawler instance and monkey-patch its fetcher
    crawler = WebCrawler("https://fakeurl.com", use_max_pages=False)
    monkeypatch.setattr(crawler, "fetcher", FakePageFetcher())

    # Run crawler and check its output structure
    crawl_results = crawler.crawl()
    assert isinstance(crawl_results, list)
    assert len(crawl_results) > 0
    for doc in crawl_results:
        assert "source" in doc
        assert "text" in doc
        assert "metadata" in doc

    # Process the crawl results through our DocumentChunkPipeline
    pipeline = DocumentChunkPipeline(
        max_tokens=1000, min_tokens=500, desired_overlap=100
    )
    final_chunks = pipeline.process(crawl_results)

    # Verify final_chunks structure
    assert isinstance(final_chunks, list)
    assert len(final_chunks) > 0

    # Check each final chunk meets the minimum token count
    for doc in final_chunks:
        tokens = default_tokenizer(doc["text"])
        assert len(tokens) >= 500, f"청크의 토큰 수가 500 미만입니다: {len(tokens)} tokens"

    # Check overlap: For chunks from the same source,
    # the current chunk should start with
    # the last 100 tokens of the previous chunk.
    source_groups = defaultdict(list)
    for doc in final_chunks:
        source_groups[doc["source"]].append(doc)

    for source, docs in source_groups.items():
        if len(docs) > 1:
            for i in range(1, len(docs)):
                prev_tokens = default_tokenizer(docs[i - 1]["text"])
                if len(prev_tokens) >= 100:
                    expected_overlap = " ".join(prev_tokens[-100:])
                else:
                    expected_overlap = " ".join(prev_tokens)
                actual_start = " ".join(
                    default_tokenizer(docs[i]["text"])[: len(expected_overlap.split())]
                )
                assert docs[i]["text"].startswith(expected_overlap), (
                    f"Source {source}의 청크 {i+1}에 overlap이 올바르게 적용되지 않았습니다.\n"
                    f"Expected overlap:\n{expected_overlap}\n"
                    f"Actual start:\n{actual_start}"
                )
