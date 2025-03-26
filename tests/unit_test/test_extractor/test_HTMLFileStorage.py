import os
import re
import hashlib
from docmesh.extractor.HTMLFileStorage import HTMLFileStorage


def test_save_html_basic(tmp_path):
    """
    save_html 메서드가 HTML 상단에 원본 URL 주석을 삽입하고,
    파일명을 "timestamp_uuidshort_md5short.html" 형태로 만들며,
    지정된 디렉토리에 파일을 제대로 생성하는지 확인.
    """
    # 1) 임시 디렉토리 생성 및 HTMLFileStorage 인스턴스 생성
    save_dir = tmp_path / "storage"
    storage = HTMLFileStorage(str(save_dir))

    # 2) 테스트용 URL, HTML
    test_url = "https://example.com/some/path?query=abc"
    test_html_content = "<html><body><p>Hello World</p></body></html>"

    # 3) save_html 호출
    saved_file_path = storage.save_html(test_url, test_html_content)

    # 4) 파일이 제대로 생성되었는지, 경로가 올바른지 확인
    assert os.path.exists(saved_file_path), "파일이 생성되지 않았습니다."
    assert saved_file_path.endswith(".html"), "파일 확장자는 .html 이어야 합니다."

    # 5) 파일 내용 검증
    with open(saved_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # (a) 상단 주석
    assert content.startswith(
        f"<!-- Original URL: {test_url} -->"
    ), "HTML 상단에 원본 URL 주석이 삽입되지 않았습니다."

    # (b) HTML 본문 포함 여부
    assert (
        "<html><body><p>Hello World</p></body></html>" in content
    ), "HTML 본문이 정상적으로 저장되지 않았습니다."

    # 6) 파일명 검증
    # 예: "20250327_113045_123456_abcd1234_abcdef12.html"
    # 실제 코드에서:
    #   timestamp = '%Y%m%d_%H%M%S_%f'
    #   random_part = uuid.uuid4().hex[:8]
    #   short_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    #   filename = f"{timestamp}_{random_part}_{short_hash}.html"
    filename = os.path.basename(saved_file_path)

    # (a) 정규 표현식으로 파일명 구조 검사
    # [날짜시각_마이크로초]_[8자리 UUID]_[8자리해시].html
    pattern = r"^\d{8}_\d{6}_\d{6}_[0-9a-f]{8}_[0-9a-f]{8}\.html$"
    assert re.match(pattern, filename), f"파일명 '{filename}'이 예상 형식과 다릅니다."

    # (b) short_hash가 올바른지 확인
    #     url을 md5 해싱 → 16진수 → 앞 8자리와 파일명의 마지막 부분 일치 여부
    expected_short_hash = hashlib.md5(test_url.encode("utf-8")).hexdigest()[:8]
    actual_short_hash = filename.split("_")[-1].split(".")[0]  # 확장자 '.' 앞 8자리
    assert (
        actual_short_hash == expected_short_hash
    ), f"파일명에 포함된 해시 {actual_short_hash}가 예상 {expected_short_hash}와 다릅니다."


def test_save_html_multiple_calls(tmp_path):
    """
    짧은 시간 간격으로 여러 번 save_html를 호출해도
    파일명이 중복되지 않고 모두 다른 파일로 저장되는지 확인.
    """
    save_dir = tmp_path / "storage"
    storage = HTMLFileStorage(str(save_dir))

    test_url = "http://example.com"
    test_html_content = "<html><body>Test</body></html>"

    file_paths = []
    for _ in range(5):
        file_path = storage.save_html(test_url, test_html_content)
        file_paths.append(file_path)

    # 파일들이 모두 달라야 함
    # 중복된 파일명이 발생했다면 하나만 생성되거나 덮어쓰였을 것
    unique_paths = set(file_paths)
    assert len(unique_paths) == 5, "짧은 간격으로 5번 호출해도 중복 없이 5개의 파일이 생성되어야 합니다."

    for path in unique_paths:
        assert os.path.exists(path), f"파일이 존재하지 않습니다: {path}"
