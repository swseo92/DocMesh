import os
import re
from typing import Optional, Tuple


class HTMLFileInspector:
    """
    - 로컬 HTML 파일을 읽어서
      1) <!-- Original URL: ... --> 형태로 삽입된 주석에서 원본 URL 추출
      2) 전체 HTML 내용을 함께 반환
    """

    @staticmethod
    def read_html(file_path: str) -> Tuple[Optional[str], str]:
        """
        파일을 열어:
        1) 원본 URL 주석 추출 (없으면 None)
        2) HTML 전체 텍스트 반환

        return: (original_url, html_content)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            contents = f.read()

        # 주석 패턴 예: <!-- Original URL: https://example.com -->
        pattern = r"<!--\s*Original URL:\s*(.*?)\s*-->"
        match = re.search(pattern, contents, flags=re.IGNORECASE)
        if match:
            original_url = match.group(1)
        else:
            original_url = None

        return (original_url, contents)
