from docmesh.extractor.tools import PageFetcher, LinkExtractor
from docmesh.extractor.HTMLFileStorage import HTMLFileStorage
from typing import List, Dict, Set
import re


class FileSavingCrawler:
    """
    - start_url부터 BFS로 링크 순회
    - HTML을 가져와서 파일로 저장
    - excluded_patterns: 정규식으로 방문 제외
    - same_domain_only: True면 시작 도메인 외부는 배제
    - max_pages: 방문할 페이지 최대 개수 (use_max_pages=True일 때만 유효)
    """

    def __init__(
        self,
        start_url: str,
        save_path: str,
        same_domain_only: bool = True,
        excluded_patterns: List[str] = None,
        use_max_pages: bool = True,
        max_pages: int = 5,
    ):
        self.start_url = start_url
        self.fetcher = PageFetcher()
        self.link_extractor = LinkExtractor()
        self.storage = HTMLFileStorage(save_path)

        self.same_domain_only = same_domain_only
        self.excluded_patterns = excluded_patterns if excluded_patterns else []
        self.use_max_pages = use_max_pages
        self.max_pages = max_pages

        self.visited: Set[str] = set()
        self.to_visit: List[str] = [start_url]

        # 최종 수집 결과: (url, file_path) 형태
        self.results: List[Dict] = []

    def crawl(self, num_batch=5) -> List[Dict]:
        """
        BFS로 URL 순회:
          1) fetch HTML
          2) 파일에 저장
          3) extract links → to_visit에 추가
          4) max_pages 확인
        모든 방문이 끝나면 self.results 반환
        """
        self.results = list()

        while True:
            if len(self.to_visit) == 0:
                break

            url = self.to_visit.pop(0)
            if url in self.visited:
                print(url)
                continue

            fetch_result = self.fetcher.fetch(url)
            if not fetch_result:
                continue

            html, content_type, final_url = fetch_result
            if final_url in self.visited:
                # 리다이렉트로 인한 중복 url 수집 방지
                continue

            self.visited.add(final_url)
            # HTML이 아닌 경우 무시
            if "text/html" not in content_type.lower():
                continue

            # 1) HTML 파일 저장
            saved_path = self.storage.save_html(url, html)
            self.results.append({"url": url, "file_path": saved_path})

            # 2) 링크 추출
            links = self.link_extractor.extract_links(html, url, self.same_domain_only)
            # excluded_patterns 필터링
            filtered = [
                link
                for link in links
                if not any(re.search(pat, link) for pat in self.excluded_patterns)
            ]
            for link in filtered:
                if link not in self.visited and link not in self.to_visit:
                    self.to_visit.append(link)

            # 3) 페이지 최대 개수 검사
            if self.use_max_pages and len(self.visited) >= self.max_pages:
                break

            if len(self.results) == num_batch:
                break

        return self.results
