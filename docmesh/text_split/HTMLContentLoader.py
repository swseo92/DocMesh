from docmesh.schema import Document
from langchain.text_splitter import HTMLHeaderTextSplitter
from typing import List


class HTMLContentLoader:
    def __init__(self):
        pass

    def load(self, url: str, html: str) -> List[Document]:
        splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "h1"),
                ("h2", "h2"),
                ("h3", "h3"),
                ("h4", "h4"),
                ("h5", "h5"),
                ("h6", "h6"),
            ]
        )
        chunks = splitter.split_text(html)
        documents = []
        for chunk in chunks:
            if isinstance(chunk, Document):
                chunk.metadata["source"] = url
                documents.append(chunk)
            else:
                documents.append(Document(page_content=chunk, metadata={"source": url}))
        return documents
