from langchain.schema import Document
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain.document_loaders import WebBaseLoader  # fallback 로더 추가
from typing import Optional, List, Tuple, Dict
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import re


class PageFetcher:
    def fetch(self, url: str) -> Optional[Tuple[str, str]]:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return (resp.text, resp.headers.get("Content-Type", ""))
        except Exception as e:
            print(f"[Fetch Error] {url}: {e}")
        return None


class LinkExtractor:
    def extract_links(
        self, html: str, base_url: str, same_domain_only: bool = True
    ) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(base_url).netloc
        links: List[str] = []
        for tag in soup.find_all("a", href=True):
            full_url = urljoin(base_url, tag["href"])
            if same_domain_only and urlparse(full_url).netloc != base_domain:
                continue
            links.append(full_url)
        return links


class LinkQueueManager:
    def __init__(self, start_url: str):
        self._start_url = start_url
        self.to_visit: List[str] = [start_url]
        self.visited: set[str] = set()

    def has_next(self) -> bool:
        return len(self.to_visit) > 0

    def next(self) -> str:
        url = self.to_visit.pop(0)
        self.visited.add(url)
        return url

    def add_links(self, links: List[str]) -> None:
        for link in links:
            if link not in self.visited and link not in self.to_visit:
                self.to_visit.append(link)


class HTMLContentLoader:
    def __init__(self, url: str, html: str):
        self.url = url
        self.html = html

    def load(self) -> List[Document]:
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
        chunks = splitter.split_text(self.html)
        documents = []
        for chunk in chunks:
            if isinstance(chunk, Document):
                chunk.metadata["source"] = self.url
                documents.append(chunk)
            else:
                documents.append(
                    Document(page_content=chunk, metadata={"source": self.url})
                )
        return documents


class WebCrawler:
    def __init__(
        self,
        start_url: str,
        use_max_pages: bool = True,
        max_pages: int = 5,
        excluded_patterns: Optional[List[str]] = None,  # 정규표현식 패턴 리스트
    ):
        self.start_url: str = start_url
        self.use_max_pages: bool = use_max_pages
        self.max_pages: int = max_pages
        # excluded_patterns가 지정되지 않으면 빈 리스트로 초기화합니다.
        self.excluded_patterns: List[str] = (
            excluded_patterns if excluded_patterns else []
        )
        self.fetcher = PageFetcher()
        self.link_extractor = LinkExtractor()
        self.queue = LinkQueueManager(start_url)
        self.collected: List[Dict] = []

        self.list_last_url = []

    def crawl(self, num_batch=None) -> List[Dict]:
        self.list_last_url = list()
        num_browsed = 0
        self.collected = list()

        while self.queue.has_next():
            url = self.queue.next()
            num_browsed += 1

            fetch_result = self.fetcher.fetch(url)
            if fetch_result is None:
                continue
            html, content_type = fetch_result

            if "text/html" not in content_type.lower():
                continue

            loader = HTMLContentLoader(url, html)
            docs = loader.load()

            if not docs:
                try:
                    fallback_loader = WebBaseLoader(url)
                    docs = fallback_loader.load()
                    print(f"[Fallback] WebBaseLoader 사용: {url}")
                except Exception as e:
                    print(f"[Fallback Error] {url}: {e}")
                    docs = []

            if not docs:
                continue

            for doc in docs:
                text = doc.page_content.strip()
                if not text:
                    continue
                self.collected.append(
                    {"source": url, "text": text, "metadata": doc.metadata}
                )

            # 하위 링크 추출
            links = self.link_extractor.extract_links(html, url)
            # 제외할 URL 패턴을 검사: 각 링크에 대해, excluded_patterns의 정규표현식과 일치하면 제외
            filtered_links = [
                link
                for link in links
                if not any(
                    re.search(pattern, link) for pattern in self.excluded_patterns
                )
            ]
            self.queue.add_links(filtered_links)
            self.list_last_url.append(url)

            if self.use_max_pages and len(self.queue.visited) >= self.max_pages:
                break

            if num_batch is not None and num_browsed == num_batch:
                return self.collected

        return self.collected


def main():
    # 예: 특정 URL와 그 하위 URL들을 제외 처리
    excluded = ["https://example.com/exclude"]
    crawler = WebCrawler(
        "https://example.com", use_max_pages=False, excluded_urls=excluded
    )
    results = crawler.crawl()
    for result in results:
        print(
            f"\n[Source] {result['source']}\nText excerpt: {result['text'][:200]}...\n"
        )


if __name__ == "__main__":
    main()
