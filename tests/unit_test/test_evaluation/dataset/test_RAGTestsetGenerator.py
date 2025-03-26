import os
import json
from unittest.mock import patch, MagicMock

from docmesh.evaluation.dataset.RAGDatasetGenerator import RAGTestsetGenerator

from docmesh.llm import BaseLLM
from docmesh.embedding import BaseEmbeddingModel
from langchain.schema import Document


@patch("glob.glob")
@patch("ragas.testset.TestsetGenerator.generate_with_langchain_docs")
@patch("langchain.document_loaders.BSHTMLLoader.load")
def test_create_testset(mock_load, mock_generate, mock_glob, tmp_path):
    """
    create_testset 메서드를 테스트합니다.
    - glob.glob은 HTML 파일 경로를 반환한다고 가정
    - BSHTMLLoader.load는 Document 리스트를 반환하도록 mock
    - TestsetGenerator.generate_with_langchain_docs의 결과를 mock
    """

    # 1) glob.glob이 반환할 테스트 HTML 파일 리스트를 지정
    mock_glob.return_value = ["test1.html", "test2.html"]

    # 2) BSHTMLLoader.load가 반환할 가짜 Document
    mock_load.return_value = [Document(page_content="Mock content", metadata={})]

    # 3) TestsetGenerator.generate_with_langchain_docs가 반환할 결과를 지정
    mock_generate.return_value = [{"question": "Q1", "answer": "A1"}]

    # 4) RAGTestsetGenerator를 초기화하기 위해 필요한 mock LLM, Embeddings 준비
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.llm = MagicMock()
    mock_embeddings = MagicMock(spec=BaseEmbeddingModel)
    mock_embeddings.embeddings = MagicMock()

    # 5) 테스트 대상 클래스 인스턴스 생성
    generator = RAGTestsetGenerator(
        directory_path=str(tmp_path),  # tmp_path는 pytest에서 제공하는 임시 디렉토리
        generator_llm=mock_llm,
        generator_embeddings=mock_embeddings,
    )

    # 6) create_testset 호출
    testset_size = 1
    result = generator.create_testset(testset_size=testset_size, num_file=2)

    # 7) 결과 검증
    assert len(result) == 1
    assert result[0]["question"] == "Q1"
    assert result[0]["answer"] == "A1"

    # 8) mock 호출 확인
    mock_glob.assert_called_once()  # glob이 한 번 호출되었는지
    mock_load.assert_called()  # BSHTMLLoader.load 호출
    mock_generate.assert_called_once()


@patch("glob.glob")
@patch("ragas.testset.TestsetGenerator.generate_with_langchain_docs")
@patch("langchain.document_loaders.BSHTMLLoader.load")
def test_create_testset_and_save(mock_load, mock_generate, mock_glob, tmp_path):
    """
    create_testset_and_save 메서드를 테스트합니다.
    실제 파일에 JSON 형태로 저장되는지 여부를 확인합니다.
    """

    # 1) mock 설정
    mock_glob.return_value = ["test1.html"]
    mock_load.return_value = [Document(page_content="Mock content", metadata={})]
    mock_generate.return_value = [{"question": "Q1", "answer": "A1"}]

    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.llm = MagicMock()
    mock_embeddings = MagicMock(spec=BaseEmbeddingModel)
    mock_embeddings.embeddings = MagicMock()

    generator = RAGTestsetGenerator(
        directory_path=str(tmp_path),
        generator_llm=mock_llm,
        generator_embeddings=mock_embeddings,
    )

    # 2) 저장할 경로와 testset_size 지정
    save_path = str(tmp_path / "testset.json")
    testset_size = 1

    # 3) create_testset_and_save 호출
    result = generator.create_testset_and_save(save_path, testset_size)

    # 4) 파일 생성 여부 확인
    assert os.path.exists(save_path), "JSON 파일이 생성되지 않았습니다."

    # 5) 파일에 올바른 내용이 저장되었는지 확인
    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["question"] == "Q1"
    assert data[0]["answer"] == "A1"

    # 6) create_testset_and_save가 반환한 result와 파일 내용이 일치하는지 확인
    assert result == data


@patch("ragas.testset.TestsetGenerator.generate_with_langchain_docs")
def test_create_testset_with_real_html(mock_generate, tmp_path):
    """
    실제 HTML 파일을 생성하여 테스트합니다.
    glob.glob, BSHTMLLoader.load는 mock 처리하지 않아 실제 파일을 로딩합니다.
    그러나 ragas의 TestsetGenerator.generate_with_langchain_docs 내부 로직은
    LLM이나 Embeddings가 필요하므로 mock 처리하여 최소한의 외부 의존성만 남깁니다.
    """

    # 1) 테스트용 HTML 파일 생성
    html_content = """
    <html>
        <head><title>Test HTML</title></head>
        <body>
            <h1>Mock Title</h1>
            <p>This is a test paragraph for checking the loader.</p>
        </body>
    </html>
    """
    test_html_path = tmp_path / "test_file.html"
    with open(test_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # 2) TestsetGenerator.generate_with_langchain_docs 에서 반환할 목업 데이터 설정
    mock_generate.return_value = [
        {"question": "Q1", "answer": "A1 from real HTML test"}
    ]

    # 3) 필요한 mock LLM, Embeddings 준비
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.llm = MagicMock()
    mock_embeddings = MagicMock(spec=BaseEmbeddingModel)
    mock_embeddings.embeddings = MagicMock()

    # 4) RAGTestsetGenerator 인스턴스 생성
    generator = RAGTestsetGenerator(
        directory_path=str(tmp_path),  # 실제 디렉토리를 사용
        generator_llm=mock_llm,
        generator_embeddings=mock_embeddings,
    )

    # 5) create_testset 호출
    testset_size = 1
    result = generator.create_testset(testset_size=testset_size, num_file=1)

    # 6) 결과 검증
    assert len(result) == 1, "Testset의 길이가 예상과 다릅니다."
    assert result[0]["question"] == "Q1"
    assert result[0]["answer"] == "A1 from real HTML test"

    # 7) 실제로 Document가 로드되었는지 확인하기 위해 mock_generate 첫 인자 확인
    # generate_with_langchain_docs 의 첫 번째 인자(docs) 안에 HTML 내용이 들어있어야 합니다.
    called_docs = mock_generate.call_args[0][
        0
    ]  # generate_with_langchain_docs(docs, ...)
    assert isinstance(called_docs, list) and len(called_docs) == 1
    assert isinstance(called_docs[0], Document)
    assert "Mock Title" in called_docs[0].page_content
    assert "This is a test paragraph" in called_docs[0].page_content

    print("테스트 통과: 실제 HTML 로딩 및 Q/A 생성 mock 테스트가 성공적으로 수행되었습니다.")
