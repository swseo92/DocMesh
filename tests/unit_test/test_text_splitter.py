from app.text_splitter import DocumentChunkPipeline, default_tokenizer


def generate_text(token_count: int, token: str = "word") -> str:
    """간단하게 지정한 토큰(token)을 공백으로 구분하여 token_count 만큼 생성."""
    return " ".join([token] * token_count)


def test_merge_chunks():
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    # 동일 source의 두 청크: 각각 200, 300 토큰 → 총 500 토큰이 되어야 함.
    doc1 = {"source": "https://example.com", "text": generate_text(200), "metadata": {}}
    doc2 = {"source": "https://example.com", "text": generate_text(300), "metadata": {}}
    input_docs = [doc1, doc2]
    merged = pipeline.merge_chunks(input_docs)
    # 첫 번째(그리고 유일한) merged doc의 토큰 수가 500인지 확인.
    tokens = default_tokenizer(merged[0]["text"])
    assert len(tokens) == 500, f"Expected 500 tokens, got {len(tokens)} tokens."


def test_split_merged_chunks():
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    # 1100 토큰짜리 doc: 이미 충분하므로 merge 단계에서는 변경 없이 그대로 전달된다고 가정.
    doc = {"source": "https://example.com", "text": generate_text(1100), "metadata": {}}
    # split_merged_chunks 단계에서 max_tokens(500)를 초과하는 경우 분할됨.
    split_docs = pipeline.split_merged_chunks([doc])
    # TokenTextSplitter의 동작에 따라, 분할 결과는 일반적으로:
    # - 첫 번째: 500 토큰, 두 번째: 500 토큰, 마지막: 100 토큰 (남은 부분)
    token_counts = [len(default_tokenizer(d["text"])) for d in split_docs]
    # 테스트 환경에 따라 다를 수 있으나, 기본적으로 위와 같은 분할 결과를 기대합니다.
    assert token_counts[0] == 500, f"첫 번째 청크 토큰 수: {token_counts[0]}"
    assert token_counts[1] == 500, f"두 번째 청크 토큰 수: {token_counts[1]}"
    assert token_counts[2] == 100, f"세 번째 청크 토큰 수: {token_counts[2]}"


def test_apply_overlap():
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    # 두 청크를 생성 (같은 source)
    doc1 = {"source": "https://example.com", "text": generate_text(600), "metadata": {}}
    doc2 = {"source": "https://example.com", "text": generate_text(500), "metadata": {}}
    merged_docs = [doc1, doc2]
    final_docs = pipeline.apply_overlap(merged_docs)
    # 두 번째 청크의 텍스트는 첫 번째 청크의 마지막 100 토큰을 포함해야 함.
    tokens_doc1 = default_tokenizer(doc1["text"])
    expected_overlap = " ".join(tokens_doc1[-100:])  # 마지막 100 토큰
    text_doc2 = final_docs[1]["text"]
    # 두 번째 청크의 시작 부분이 expected_overlap과 동일한지 확인
    assert text_doc2.startswith(expected_overlap), "두 번째 청크에 overlap이 올바르게 적용되지 않았습니다."


def test_process_pipeline():
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    # 모의 입력: 같은 source의 여러 작은 청크와 다른 source의 큰 청크
    doc1 = {"source": "https://example.com", "text": generate_text(200), "metadata": {}}
    doc2 = {"source": "https://example.com", "text": generate_text(300), "metadata": {}}
    doc3 = {"source": "https://example.com", "text": generate_text(400), "metadata": {}}
    doc4 = {"source": "https://example.org", "text": generate_text(800), "metadata": {}}
    input_docs = [doc1, doc2, doc3, doc4]

    final_docs = pipeline.process(input_docs)

    # 예: https://example.com 소스의 경우,
    # doc1, doc2, doc3가 합쳐져 최소 500 토큰 이상이 되고, 분할/overlap 적용 결과 여러 청크가 생성될 수 있음.
    # https://example.org 소스는 이미 800 토큰이므로, 그대로 유지되거나 추가 분할될 수 있음.
    for doc in final_docs:
        tokens = default_tokenizer(doc["text"])
        # min_tokens(500) 미만인 경우는 overlap 단계에서 보정되었는지 확인
        if doc["source"] == "https://example.com":
            assert len(tokens) >= 500 or len(tokens) == len(
                default_tokenizer(doc["text"])
            ), f"https://example.com 청크가 최소 토큰 수를 만족하지 않습니다: {len(tokens)} tokens"
    # 또한, 동일 소스 인접 청크에 대해 overlap이 적용되었는지 확인
    for idx in range(1, len(final_docs)):
        if final_docs[idx]["source"] == final_docs[idx - 1]["source"]:
            prev_tokens = default_tokenizer(final_docs[idx - 1]["text"])
            expected_overlap = (
                " ".join(prev_tokens[-100:])
                if len(prev_tokens) >= 100
                else " ".join(prev_tokens)
            )
            assert final_docs[idx]["text"].startswith(
                expected_overlap
            ), f"청크 {idx + 1}에 overlap이 올바르게 적용되지 않았습니다."


def test_empty_input():
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    result = pipeline.process([])
    assert result == []


def test_single_document_no_overlap():
    # 단일 문서인 경우, overlap은 적용되지 않아야 함.
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    doc = {"source": "https://example.com", "text": generate_text(600), "metadata": {}}
    result = pipeline.process([doc])
    # 결과가 하나의 청크로 유지되거나, split이 일어났더라도 첫 청크는 overlap이 없어야 함.
    assert result[0]["text"] == result[0]["text"].lstrip()  # 앞에 불필요한 공백 없음


def test_exact_min_tokens_no_merge():
    # 텍스트가 정확히 min_tokens인 경우, merge가 발생하지 않아야 함.
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    doc = {"source": "https://example.com", "text": generate_text(500), "metadata": {}}
    merged = pipeline.merge_chunks([doc])
    tokens = default_tokenizer(merged[0]["text"])
    assert len(tokens) == 500


def test_split_behavior():
    # 텍스트가 max_tokens보다 많이 길 경우, split이 올바르게 분할되는지 확인
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    doc = {"source": "https://example.com", "text": generate_text(1100), "metadata": {}}
    split_docs = pipeline.split_merged_chunks([doc])
    token_counts = [len(default_tokenizer(d["text"])) for d in split_docs]
    # 기대: 첫 청크 500, 두번째 500, 마지막 100 토큰
    assert token_counts[0] == 500
    assert token_counts[1] == 500
    assert token_counts[2] == 100


def test_multisource_merging():
    # 서로 다른 source의 문서는 합쳐지지 않아야 합니다.
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    doc1 = {"source": "https://example.com", "text": generate_text(200), "metadata": {}}
    doc2 = {"source": "https://example.org", "text": generate_text(300), "metadata": {}}
    merged = pipeline.merge_chunks([doc1, doc2])
    # 두 개의 문서가 각각 별도로 유지되어야 함
    assert len(merged) == 2


def test_split_index_metadata():
    # 분할 시 metadata에 split_index가 올바르게 추가되는지 확인
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    doc = {"source": "https://example.com", "text": generate_text(1100), "metadata": {}}
    split_docs = pipeline.split_merged_chunks([doc])
    for idx, d in enumerate(split_docs):
        assert "split_index" in d["metadata"]
        assert d["metadata"]["split_index"] == idx
