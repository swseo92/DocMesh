from docmesh.text_splitter import (
    DocumentChunkPipeline,
    default_tokenizer,
)


def generate_text(token_count: int, token: str = "word") -> str:
    """지정한 토큰(token)을 공백으로 구분하여 token_count 만큼 생성."""
    return " ".join([token] * token_count)


def test_merge_chunks():
    pipeline = DocumentChunkPipeline(
        max_tokens=1000, min_tokens=500, desired_overlap=100
    )
    # 동일 source의 두 청크: 각각 200, 300 토큰 → 총 500 토큰이 되어야 함.
    doc1 = {"source": "https://example.com", "text": generate_text(200), "metadata": {}}
    doc2 = {"source": "https://example.com", "text": generate_text(300), "metadata": {}}
    input_docs = [doc1, doc2]
    merged = pipeline.merge_chunks(input_docs)
    tokens = default_tokenizer(merged[0]["text"])
    assert len(tokens) == 500, f"Expected 500 tokens, got {len(tokens)} tokens."


def test_split_merged_chunks():
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    # 1100 토큰짜리 doc: split 단계에서 500, 500, 100 토큰으로 분할되어야 함.
    doc = {"source": "https://example.com", "text": generate_text(1100), "metadata": {}}
    split_docs = pipeline.split_merged_chunks([doc])
    token_counts = [len(default_tokenizer(d["text"])) for d in split_docs]
    assert token_counts[0] == 500, f"첫 번째 청크 토큰 수: {token_counts[0]}"
    assert token_counts[1] == 500, f"두 번째 청크 토큰 수: {token_counts[1]}"
    assert token_counts[2] == 100, f"세 번째 청크 토큰 수: {token_counts[2]}"


def test_apply_overlap():
    pipeline = DocumentChunkPipeline(
        max_tokens=1000, min_tokens=500, desired_overlap=100
    )
    # 동일 source의 두 청크 생성
    doc1 = {"source": "https://example.com", "text": generate_text(600), "metadata": {}}
    doc2 = {"source": "https://example.com", "text": generate_text(500), "metadata": {}}
    merged_docs = [doc1, doc2]
    final_docs_dict = pipeline.apply_overlap(merged_docs)
    # final_docs_dict는 여전히 dict 리스트이므로, 토큰 리스트로 비교합니다.
    tokens_doc1 = default_tokenizer(doc1["text"])
    expected_overlap_tokens = tokens_doc1[-100:]  # 마지막 100 토큰
    tokens_doc2 = default_tokenizer(final_docs_dict[1]["text"])
    # 두 번째 청크의 앞부분 100 토큰과 비교
    assert tokens_doc2[:100] == expected_overlap_tokens, "Overlap이 올바르게 적용되지 않았습니다."


def test_process_pipeline():
    pipeline = DocumentChunkPipeline(
        max_tokens=1000, min_tokens=500, desired_overlap=100
    )
    # 모의 입력: 여러 청크가 포함된 문서들 (서로 같은 source와 다른 source)
    doc1 = {"source": "https://example.com", "text": generate_text(200), "metadata": {}}
    doc2 = {"source": "https://example.com", "text": generate_text(300), "metadata": {}}
    doc3 = {"source": "https://example.com", "text": generate_text(400), "metadata": {}}
    doc4 = {"source": "https://example.org", "text": generate_text(800), "metadata": {}}
    input_docs = [doc1, doc2, doc3, doc4]
    final_docs = pipeline.process(input_docs)
    # final_docs는 LC_Document 객체 리스트
    for doc in final_docs:
        tokens = default_tokenizer(doc.page_content)
        if doc.metadata.get("source") == "https://example.com":
            assert (
                len(tokens) >= 500
            ), f"https://example.com 청크가 최소 토큰 수 미만: {len(tokens)} tokens"
    # 동일 소스 인접 청크 간 overlap 검증: LC_Document의 page_content를 토큰화하여 비교
    for idx in range(1, len(final_docs)):
        if final_docs[idx].metadata.get("source") == final_docs[idx - 1].metadata.get(
            "source"
        ):
            prev_tokens = default_tokenizer(final_docs[idx - 1].page_content)
            expected_overlap = (
                prev_tokens[-100:] if len(prev_tokens) >= 100 else prev_tokens
            )
            current_tokens = default_tokenizer(final_docs[idx].page_content)
            assert (
                current_tokens[: len(expected_overlap)] == expected_overlap
            ), f"청크 {idx+1}의 overlap이 올바르지 않습니다."


def test_empty_input():
    pipeline = DocumentChunkPipeline(
        max_tokens=1000, min_tokens=500, desired_overlap=100
    )
    result = pipeline.process([])
    assert result == []


def test_single_document_no_overlap():
    # 단일 문서인 경우, overlap은 적용되지 않아야 함.
    pipeline = DocumentChunkPipeline(
        max_tokens=1000, min_tokens=500, desired_overlap=100
    )
    doc_text = generate_text(600)
    doc = {"source": "https://example.com", "text": doc_text, "metadata": {}}
    result = pipeline.process([doc])
    # 단일 Document이면 page_content는 원본과 동일
    assert result[0].page_content.strip() == doc_text


def test_exact_min_tokens_no_merge():
    # 텍스트가 정확히 min_tokens인 경우, merge가 발생하지 않아야 함.
    pipeline = DocumentChunkPipeline(
        max_tokens=1000, min_tokens=500, desired_overlap=100
    )
    doc = {"source": "https://example.com", "text": generate_text(500), "metadata": {}}
    merged = pipeline.merge_chunks([doc])
    tokens = default_tokenizer(merged[0]["text"])
    assert len(tokens) == 500


def test_split_behavior():
    # 텍스트가 max_tokens보다 긴 경우, split이 올바르게 분할되는지 확인
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    doc = {"source": "https://example.com", "text": generate_text(1100), "metadata": {}}
    split_docs = pipeline.split_merged_chunks([doc])
    token_counts = [len(default_tokenizer(d["text"])) for d in split_docs]
    assert token_counts[0] == 500
    assert token_counts[1] == 500
    assert token_counts[2] == 100


def test_multisource_merging():
    # 서로 다른 source의 문서는 merge되어서는 안 됩니다.
    pipeline = DocumentChunkPipeline(
        max_tokens=1000, min_tokens=500, desired_overlap=100
    )
    doc1 = {"source": "https://example.com", "text": generate_text(200), "metadata": {}}
    doc2 = {"source": "https://example.org", "text": generate_text(300), "metadata": {}}
    merged = pipeline.merge_chunks([doc1, doc2])
    assert len(merged) == 2


def test_split_index_metadata():
    # split 시 metadata에 split_index가 올바르게 추가되는지 확인
    pipeline = DocumentChunkPipeline(
        max_tokens=1000, min_tokens=500, desired_overlap=100
    )
    doc = {"source": "https://example.com", "text": generate_text(1100), "metadata": {}}
    split_docs = pipeline.split_merged_chunks([doc])
    for idx, d in enumerate(split_docs):
        assert "split_index" in d["metadata"]
        assert d["metadata"]["split_index"] == idx
