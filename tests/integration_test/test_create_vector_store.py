import os
import glob
import shutil

from docmesh.extractor.HTMLFileInspector import HTMLFileInspector
from docmesh.text_split import HTMLContentLoader
from docmesh.text_split.DocumentMerger import DocumentMerger
from docmesh.text_split.DocumentChunkSplitter import DocumentChunkSplitter
from docmesh.embedding import EmbeddingModelFactory
from docmesh.vector_store import VectorStoreFactory

from dotenv import load_dotenv


def test_create_and_save_and_load_vector_store():
    load_dotenv()

    path_dir_test_html = "../../test_data/html"
    list_fpath_test_html = glob.glob(path_dir_test_html + "/*.html")[:2]
    assert len(list_fpath_test_html) == 2

    path_save_faiss = "./faiss_index"
    if os.path.isdir(path_save_faiss):
        shutil.rmtree(path_save_faiss)
    assert not os.path.exists(path_save_faiss)

    # 파일 로드 객체 생성
    html_file_loader = HTMLFileInspector()
    html_content_loader = HTMLContentLoader()

    # 전처리 객체 생성
    doc_merger = DocumentMerger(min_tokens=500)
    doc_splitter = DocumentChunkSplitter(chunk_size=1000, chunk_overlap=100)
    # EmbeddingModelFactory를 통해 임베딩 모델 생성
    embedding_model = EmbeddingModelFactory().create_embedding_model(
        provider="openai", model_name="text-embedding-ada-002"
    )
    # VectorStoreFactory를 통해 벡터 스토어 생성
    vector_store = VectorStoreFactory().create_vector_store(
        provider="faiss",
        embedding_model=embedding_model,
    )

    url, html = html_file_loader.read_html(list_fpath_test_html[0])
    docs_langchain = html_content_loader.load(url, html)

    docs_merged = doc_merger.merge_documents(docs_langchain)  # 최소 토큰 기준으로 chunk 병합
    docs_splitted = doc_splitter.chunkify(
        docs_merged
    )  # 최대 토큰 기준으로 chunk 분리 및 overlap 생성

    vector_store.add_documents(docs_splitted)
    vector_store.save_local(path_save_faiss)

    # 저장이 정상적으로 동작하는지 확인
    assert os.path.exists(path_save_faiss)

    vector_store2 = VectorStoreFactory().create_vector_store(
        provider="faiss", embedding_model=embedding_model, path=path_save_faiss
    )

    # 로드한 vector store와 기존 vecotr store 결과 값 비교
    assert vector_store.similarity_search("qux") == vector_store2.similarity_search(
        "qux"
    )
