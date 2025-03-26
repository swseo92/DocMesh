from typing import List, Dict
import yaml

# Ragas
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona

# LangChain
from langchain.document_loaders import BSHTMLLoader
from langchain.schema import Document

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from docmesh.text_split import DocumentChunkSplitter


class RAGTestsetGenerator:
    def __init__(
        self,
        generator_llm: BaseChatModel,
        critic_llm: BaseChatModel,
        generator_embeddings: Embeddings,
        list_persona: List[Persona] = None,
        transforms: list = None,
    ):
        self.testset_gen = TestsetGenerator.from_langchain(
            generator_llm, critic_llm, generator_embeddings
        )
        self.testset_gen.persona_list = list_persona
        self.transforms = transforms

    def _load_html_documents(self, list_fpath: list[str]) -> List[Document]:
        docs = []

        i = 0
        for fpath in list_fpath:
            loader = BSHTMLLoader(fpath, "utf-8")
            loaded = loader.load()  # -> List[Document]
            # 문서별로 metadata에 "source_file" 등을 추가해도 좋음
            docs.extend(loaded)
            i += 1
        return docs

    def create_testset(
        self, list_fpath: list[str], testset_size: int, transforms: list = None
    ) -> List[Dict]:
        if transforms is None:
            transforms = self.transforms
        # 1) 문서 로딩
        docs = self._load_html_documents(list_fpath)

        splitter = DocumentChunkSplitter(chunk_size=1000, chunk_overlap=100)
        docs_splitted = splitter.chunkify(docs)

        # 2) Ragas TestsetGenerator로부터 Q/A 생성
        testset = self.testset_gen.generate_with_langchain_docs(
            docs_splitted, testset_size=testset_size, transforms=transforms
        )
        return testset


def load_personas_from_yaml(yaml_file_path: str) -> List[Persona]:
    """
    주어진 YAML 파일에서 Persona 목록을 파싱해 반환합니다.
    :param yaml_file_path: YAML 파일 경로
    :return: ragas.testset.persona.Persona 객체들의 리스트
    """
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  # YAML 내용을 파이썬 객체(보통 list[dict])로 파싱

    persona_list = []
    for item in data["evaluation"]["dataset"]["personas"]:
        persona_obj = Persona(
            name=item["name"], role_description=item["role_description"]
        )
        persona_list.append(persona_obj)

    return persona_list


# 사용 예시
if __name__ == "__main__":
    personas = load_personas_from_yaml("personas.yaml")
    for p in personas:
        print(f"[{p.name}] {p.role_description}")
