from typing import List, Dict
import glob

# Ragas
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain
from langchain.document_loaders import BSHTMLLoader
from langchain.schema import Document

from docmesh.embedding import BaseEmbeddingModel
from docmesh.llm import BaseLLM


class RAGTestsetGenerator:
    def __init__(
        self,
        directory_path: str,
        generator_llm: BaseLLM,
        generator_embeddings: BaseEmbeddingModel,
    ):
        self.directory_path = directory_path

        generator_llm = LangchainLLMWrapper(generator_llm.llm)
        generator_embeddings = LangchainEmbeddingsWrapper(
            generator_embeddings.embeddings
        )

        self.testset_gen = TestsetGenerator(
            llm=generator_llm, embedding_model=generator_embeddings
        )

    def _load_html_documents(self, num_file=None) -> List[Document]:
        docs = []

        i = 0
        for fpath in glob.glob(self.directory_path + "/*.html"):
            loader = BSHTMLLoader(fpath, "utf-8")
            loaded = loader.load()  # -> List[Document]
            # 문서별로 metadata에 "source_file" 등을 추가해도 좋음
            docs.extend(loaded)
            i += 1
            if i >= num_file:
                break
        return docs

    def create_testset(self, testset_size: int, num_file=5) -> List[Dict]:
        # 1) 문서 로딩
        docs = self._load_html_documents(num_file=5)

        # 2) Ragas TestsetGenerator로부터 Q/A 생성
        testset = self.testset_gen.generate_with_langchain_docs(
            docs, testset_size=testset_size
        )

        return testset

    def create_testset_and_save(self, save_path: str, testset_size: int) -> List[Dict]:
        import json

        testset = self.create_testset(testset_size)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(testset, f, indent=2, ensure_ascii=False)
        return testset
