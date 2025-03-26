from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docmesh.config import Config


class DocumentChunkSplitter:
    """
    - LangChain Document 리스트를 받아,
      지정된 chunk_size, chunk_overlap 기준으로
      각 문서를 여러 chunk Document로 분할하는 클래스.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        encoding_name: str = Config.TIKTOKEN_ENCODING,
    ):
        """
        :param chunk_size: 한 청크의 최대 토큰 수
        :param chunk_overlap: 청크 간 겹쳐질 토큰 수
        :param length_function: 텍스트 길이를 계산하는 함수 (기본: 공백 단위)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name

        # LangChain의 TokenTextSplitter 사용
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=self.encoding_name,
        )

    def chunkify(self, docs: List[Document]) -> List[Document]:
        """
        각 Document(page_content)를 splitter로 분할하고,
        분할된 텍스트마다 새 Document를 생성.
        metadata는 원본 metadata를 복사하고,
        split_index 같은 추가 정보를 부여할 수도 있음.
        """
        result_docs: List[Document] = []

        for doc in docs:
            text = doc.page_content
            original_metadata = doc.metadata.copy()

            # splitter.split_text() -> List[str]
            splitted_texts = self.splitter.split_text(text)

            if len(splitted_texts) <= 1:
                # 분할 결과가 1개 이하라면 그냥 원본 doc 그대로 추가
                result_docs.append(doc)
            else:
                # 여러 chunk로 나뉜 경우
                for idx, chunk_text in enumerate(splitted_texts):
                    # 새 Document를 생성하면서, split_index를 부여
                    new_metadata = original_metadata.copy()
                    new_metadata["split_index"] = idx

                    new_doc = Document(page_content=chunk_text, metadata=new_metadata)
                    result_docs.append(new_doc)

        return result_docs
