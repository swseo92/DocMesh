import hashlib
import urllib.parse
import re
import os
from typing import Optional


class HTMLFileStorage:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_html(self, url: str, html: str) -> str:
        """
        HTML 상단에 <!-- Original URL: ... --> 주석을 삽입하여,
        파일 단독으로도 원본 URL을 복원 가능하도록 저장.
        """

        # 1) HTML 최상단에 원본 URL 주석 추가
        annotated_html = f"<!-- Original URL: {url} -->\n{html}"

        # 2) URL을 파일명으로 활용할 때, 안전하게 인코딩/해시 처리 (예시)
        #    - 너무 긴 파일명을 방지하기 위해 MD5 해시의 앞 8자리 사용
        safe_url_str = urllib.parse.quote_plus(url)
        short_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
        filename = f"{safe_url_str}_{short_hash}.html"

        # 3) 파일에 저장
        file_path = os.path.join(self.save_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(annotated_html)

        return file_path


def extract_original_url(file_path: str) -> Optional[str]:
    """
    주어진 HTML 파일에서 <!-- Original URL: ... --> 형태의 주석을 찾아
    원본 URL을 추출해 반환합니다.

    - file_path: HTML 파일 경로
    - return: 추출된 URL 문자열 (찾지 못하면 None)
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # HTML 파일 전체를 읽어옵니다. (상단 몇 줄만 읽어도 되지만, 안전을 위해 전체 검색)
    with open(file_path, "r", encoding="utf-8") as f:
        contents = f.read()

    # 주석 패턴: <!-- Original URL: (URL) -->
    # 괄호 내에 임의의 문자(.*?)를 *비탐욕적으로* 매칭하여 URL을 추출
    pattern = r"<!--\s*Original URL:\s*(.*?)\s*-->"
    match = re.search(pattern, contents, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None
