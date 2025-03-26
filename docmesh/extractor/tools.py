from langchain.schema import Document
from langchain.text_splitter import HTMLHeaderTextSplitter
from typing import Optional, List, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin


def normalize_url(
    url: str,
    force_https: bool = False,
    remove_www: bool = False,
    remove_fragment: bool = True,
    remove_trailing_slash: bool = False,
    remove_query_params_prefixes: tuple = ("utm_", "fbclid"),
) -> str:
    """
    URL을 정규화하여, 중복 방지용으로 사용하기 위한 예시 함수.

    Parameters
    ----------
    url : str
        대상 URL
    force_https : bool
        True면 http -> https로 강제 변경 (기본 False)
    remove_www : bool
        True면 도메인 앞의 'www.' 제거
    remove_fragment : bool
        True면 # 이하 제거
    remove_trailing_slash : bool
        True면 path 끝의 '/' 제거 (단, 루트 '/'는 남김)
    remove_query_params_prefixes : tuple
        특정 접두사로 시작하는 쿼리 파라미터를 제거 (utm_, fbclid 등)

    Returns
    -------
    str
        정규화된 URL 문자열
    """
    # 1) 공백 제거
    url = url.strip()

    # 2) urlparse
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc
    path = parsed.path
    query = parsed.query
    fragment = parsed.fragment

    # 3) force https
    if force_https and scheme in ["http", "https"]:
        scheme = "https"

    # 4) 도메인(호스트) 소문자화
    netloc = netloc.lower()

    # 5) www 제거
    if remove_www and netloc.startswith("www."):
        netloc = netloc[4:]

    # 6) 쿼리 파라미터 필터링
    #    예: utm_*, fbclid 등 불필요 파라미터 제거
    if query:
        # parse_qsl -> List of (key, value)
        qparams = parse_qsl(query, keep_blank_values=True)
        filtered_qparams = []
        for k, v in qparams:
            # 특정 접두사(prefix)로 시작하면 제외
            if any(k.startswith(prefix) for prefix in remove_query_params_prefixes):
                continue
            filtered_qparams.append((k, v))
        query = urlencode(filtered_qparams)

    # 7) 프래그먼트 제거
    if remove_fragment:
        fragment = ""

    # 8) 트레일링 슬래시 제거
    #    (단, 경로가 "/" 하나만 있으면 루트이므로 그대로 둠)
    if remove_trailing_slash and len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # 9) urlunparse 로 재조합
    normalized = urlunparse((scheme, netloc, path, parsed.params, query, fragment))

    return normalized


class PageFetcher:
    def fetch(self, url: str) -> Optional[Tuple[str, str]]:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                final_url = resp.url
                return (resp.text, resp.headers.get("Content-Type", ""), final_url)
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
            # 동일 도메인만 수집
            if same_domain_only and urlparse(full_url).netloc != base_domain:
                continue

            url_normalized = normalize_url(full_url)
            links.append(url_normalized)
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
            link_refined = normalize_url(link)
            if link_refined not in self.visited and link_refined not in self.to_visit:
                self.to_visit.append(link_refined)


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
