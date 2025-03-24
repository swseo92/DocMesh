from langchain.schema import Document
from langchain.text_splitter import HTMLHeaderTextSplitter
from typing import Optional, List, Tuple, Dict
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup


# 실제 페이지의 HTML을 가져오는 클래스 (한 번만 호출)
class PageFetcher:
    def fetch(self, url: str) -> Optional[Tuple[str, str]]:
        """
        주어진 URL에서 HTML 콘텐츠를 가져오며,
        (html, Content-Type)를 튜플로 반환.
        실패 시 None 반환.
        """
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return (resp.text, resp.headers.get("Content-Type", ""))
        except Exception as e:
            print(f"[Fetch Error] {url}: {e}")
        return None


# HTML에서 하이퍼링크(a href)를 추출하는 클래스
class LinkExtractor:
    def extract_links(
        self, html: str, base_url: str, same_domain_only: bool = True
    ) -> List[str]:
        """
        HTML 내 모든 링크를 추출합니다.
        same_domain_only=True이면, base_url과 같은 도메인의 링크만 포함.
        """
        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(base_url).netloc
        links: List[str] = []
        for tag in soup.find_all("a", href=True):
            full_url = urljoin(base_url, tag["href"])
            if same_domain_only and urlparse(full_url).netloc != base_domain:
                continue
            links.append(full_url)
        return links


# 방문한 URL과 방문 예정 URL을 관리하는 큐 클래스
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


# HTMLContentLoader: HTMLHeaderTextSplitter를 사용해 HTML을 여러 Document(청크)로 분할
class HTMLContentLoader:
    def __init__(self, url: str, html: str):
        self.url = url
        self.html = html

    def load(self) -> List[Document]:
        # headers_to_split_on에 튜플 형태로 헤더 태그 정보를 전달합니다.
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
        # splitter가 이미 Document 객체를 반환하는 경우, 메타데이터를 업데이트하고,
        # 그렇지 않으면 새 Document를 생성합니다.
        for chunk in chunks:
            if isinstance(chunk, Document):
                chunk.metadata["source"] = self.url
                documents.append(chunk)
            else:
                documents.append(
                    Document(page_content=chunk, metadata={"source": self.url})
                )
        return documents


# 전체 웹 크롤링 흐름을 조율하는 메인 클래스
class WebCrawler:
    def __init__(self, start_url: str, use_max_pages: bool = True, max_pages: int = 5):
        self.start_url: str = start_url
        self.use_max_pages: bool = use_max_pages
        self.max_pages: int = max_pages
        self.fetcher = PageFetcher()
        self.link_extractor = LinkExtractor()
        self.queue = LinkQueueManager(start_url)
        # 메타데이터를 포함한 청크를 딕셔너리 형태로 저장
        self.collected: List[Dict] = []

    def crawl(self) -> List[Dict]:
        """
        BFS 방식으로 크롤링을 수행합니다.
        각 URL에 대해 PageFetcher로 HTML을 한 번만 가져와서,
          - HTMLContentLoader로 Document 객체(텍스트 청크)를 생성하고
          - LinkExtractor로 하위 링크를 추출합니다.
        각 청크는 개별적으로 메타데이터와 함께 저장됩니다.
        """
        while self.queue.has_next():
            url = self.queue.next()

            fetch_result = self.fetcher.fetch(url)
            if fetch_result is None:
                continue
            html, content_type = fetch_result

            if "text/html" not in content_type.lower():
                continue

            loader = HTMLContentLoader(url, html)
            docs = loader.load()

            if not docs:
                continue

            # 각 Document(청크)를 개별적으로 저장 (메타데이터 포함)
            for doc in docs:
                text = doc.page_content.strip()
                if not text:
                    continue
                self.collected.append(
                    {"source": url, "text": text, "metadata": doc.metadata}
                )

            # 하위 링크 추출 (이미 fetch한 HTML 재사용)
            links = self.link_extractor.extract_links(html, url)
            self.queue.add_links(links)

            if self.use_max_pages and len(self.queue.visited) >= self.max_pages:
                break

        return self.collected


def main():
    crawler = WebCrawler("https://example.com", use_max_pages=False)
    results = crawler.crawl()
    for result in results:
        print(
            f"\n[Source] {result['source']}\nText excerpt: {result['text'][:200]}...\n"
        )


# 사용 예시
if __name__ == "__main__":
    main()
