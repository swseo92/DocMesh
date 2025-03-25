import os
from abc import ABC, abstractmethod


class BaseDocumentLoader(ABC):
    """
    폴더 내 파일을 순회하면서, 특정 포맷의 문서를 로딩/파싱하여
    (파일 경로, 파싱된 콘텐츠)를 제너레이터로 반환하는 추상 클래스입니다.
    """

    def __init__(self, directory_path: str, valid_extensions=None):
        """
        :param directory_path: 문서 파일들이 위치한 폴더 경로
        :param valid_extensions: 로드할 때 허용할 확장자 목록 (예: [".html", ".md", ...])
        """
        self.directory_path = directory_path
        self.valid_extensions = valid_extensions if valid_extensions else []

    def list_files(self):
        """
        directory_path 내 파일들을 순회하며,
        valid_extensions에 속하는 확장자만 필터링 후 반환.
        """
        if not os.path.isdir(self.directory_path):
            raise FileNotFoundError(f"Directory not found: {self.directory_path}")

        all_files = sorted(os.listdir(self.directory_path))
        for fname in all_files:
            # 확장자가 valid_extensions에 해당하는지 확인
            if self._check_extension(fname):
                yield os.path.join(self.directory_path, fname)

    def _check_extension(self, filename: str) -> bool:
        """
        내부용 메서드:
        valid_extensions가 비어있지 않다면 그 범위 내인지 확인,
        비어있다면 모든 파일 허용 (선택적 로직).
        """
        if not self.valid_extensions:
            return True  # 확장자 제한 없음
        _, ext = os.path.splitext(filename)
        return ext.lower() in [v.lower() for v in self.valid_extensions]

    @abstractmethod
    def parse_file(self, file_path: str) -> str:
        """
        실제 파일을 읽고, 포맷별로 필요한 전처리/파싱을 수행하여 텍스트 등을 반환.
        구체적인 로직은 하위 클래스에서 구현해야 함.
        """
        pass

    def iter_documents(self):
        """
        파일 목록을 순회하며 parse_file을 적용한 결과를
        (파일 경로, 파싱된 컨텐츠) 형태로 제너레이터로 전달.
        """
        for file_path in self.list_files():
            # 하위 클래스에서 구현한 parse_file을 통해 파싱
            parsed_content = self.parse_file(file_path)
            yield (file_path, parsed_content)
