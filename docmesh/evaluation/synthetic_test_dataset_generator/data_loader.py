import os


class HTMLLoader:
    """
    폴더 내 여러 HTML 파일을 하나씩 제너레이터 방식으로 반환하는 클래스입니다.
    """

    def __init__(self, directory_path: str):
        """
        :param directory_path: HTML 파일들이 저장된 폴더 경로
        """
        self.directory_path = directory_path

    def iter_html_contents(self):
        """
        폴더 내 모든 .html 파일에 대해, (파일 경로, 파일 내용)을 제너레이터로 반환합니다.

        usage:
        for file_path, html_text in loader.iter_html_contents():
            ...
        """
        if not os.path.isdir(self.directory_path):
            raise FileNotFoundError(f"Directory not found: {self.directory_path}")

        file_list = sorted(os.listdir(self.directory_path))  # 정렬(선택)

        for fname in file_list:
            if fname.endswith(".html"):
                file_path = os.path.join(self.directory_path, fname)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # 하나의 파일에 대해 (파일 경로, 파일 내용)을 산출
                yield (file_path, content)
