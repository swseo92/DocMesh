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
        generator_llm=mock_llm,
        critic_llm=mock_llm,
        generator_embeddings=mock_embeddings,
    )

    # 6) create_testset 호출
    testset_size = 1
    result = generator.create_testset(str(tmp_path), testset_size=testset_size)

    # 7) 결과 검증
    assert len(result) == 1
    assert result[0]["question"] == "Q1"
    assert result[0]["answer"] == "A1"

    # 8) mock 호출 확인
    mock_load.assert_called()  # BSHTMLLoader.load 호출
    mock_generate.assert_called_once()
