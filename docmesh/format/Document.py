from langchain.schema import Document as LC_Document


class Document:
    """
    Document 클래스는 임베딩 및 검색을 위한 텍스트 청크와 메타데이터를 캡슐화합니다.
    """

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

    def to_langchain_document(self) -> LC_Document:
        return LC_Document(page_content=self.page_content, metadata=self.metadata)
