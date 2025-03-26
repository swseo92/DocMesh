import hashlib
import datetime
import uuid
import os


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
        # 마이크로초까지 포함한 현재 시각

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # UUID 중 8자리만 사용 (충돌 가능성 낮춤)
        random_part = uuid.uuid4().hex[:8]

        short_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
        filename = f"{timestamp}_{random_part}_{short_hash}.html"

        # 3) 파일에 저장
        file_path = os.path.join(self.save_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(annotated_html)

        return file_path
