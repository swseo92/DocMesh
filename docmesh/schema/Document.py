from langchain.schema import Document
from typing import Optional, Dict, Any


class DocMeshDocument(Document):
    """
    - LangChain Document를 상속한 커스텀 클래스
    - page_content, metadata는 부모 그대로 사용
    - 필요한 추가 속성/메서드는 자유롭게 정의
    """

    def __init__(
        self, page_content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs
    ):
        super().__init__(page_content=page_content, metadata=metadata or {})
        # kwargs 등을 통해 추가 처리 가능
        # self.custom_attr = kwargs.get("custom_attr", None)

    @classmethod
    def from_document(cls, doc: Document) -> "DocMeshDocument":
        """
        - 만약 doc가 이미 DocMeshDocument라면 그대로 반환
        - 그렇지 않으면 DocMeshDocument로 새로 감싸서 반환
        """
        if isinstance(doc, cls):
            return doc  # 이미 DocMeshDocument 타입
        else:
            return cls(page_content=doc.page_content, metadata=doc.metadata)

    def set_source(self, source: str) -> None:
        self.metadata["source"] = source

    def get_source(self) -> Optional[str]:
        return self.metadata.get("source")

    def __repr__(self):
        return (
            f"DocMeshDocument("
            f"source={self.get_source()}, "
            f"content='{self.page_content[:30]}...', "
            f"metadata={self.metadata}"
            f")"
        )
