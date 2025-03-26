import os
import hashlib
import urllib.parse
from docmesh.extractor.HTMLFileStorage import HTMLFileStorage, extract_original_url


def test_save_html(tmp_path):
    """
    HTMLFileStorage.save_html 메서드가
    1) HTML 상단에 주석 형태로 Original URL을 추가하는지
    2) 예상된 파일 이름으로 파일을 저장하는지
    3) 저장 후 반환된 경로가 실제로 존재하는지
    등을 검증합니다.
    """

    # 1) 테스트용 디렉토리 및 인스턴스 생성
    save_dir = tmp_path / "html_files"
    storage = HTMLFileStorage(str(save_dir))

    # 2) 테스트 입력 데이터
    test_url = "http://example.com/some/path?query=test"
    test_html_content = "<html><body><p>Hello World</p></body></html>"

    # 3) save_html 호출
    saved_file_path = storage.save_html(url=test_url, html=test_html_content)

    # 4) 반환된 경로가 실제로 존재하는지 확인
    assert os.path.exists(saved_file_path), "HTML 파일이 지정된 경로에 생성되지 않았습니다."

    # 5) 파일 내용을 확인
    with open(saved_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # (a) 상단 주석 삽입 확인
    assert content.startswith(
        f"<!-- Original URL: {test_url} -->"
    ), "HTML 상단에 Original URL 주석이 삽입되지 않았습니다."

    # (b) 실제 HTML 내용이 포함되어 있는지 확인
    assert (
        "<html><body><p>Hello World</p></body></html>" in content
    ), "HTML 본문이 정상적으로 저장되지 않았습니다."

    # (c) 파일명 검사
    # 파일명은 "safe_url_str_{short_hash}.html" 형태여야 함
    # safe_url_str = urllib.parse.quote_plus(url)
    # short_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    # filename = f"{safe_url_str}_{short_hash}.html"
    safe_url_str = urllib.parse.quote_plus(test_url)
    short_hash = hashlib.md5(test_url.encode("utf-8")).hexdigest()[:8]
    expected_filename = f"{safe_url_str}_{short_hash}.html"
    expected_path = os.path.join(str(save_dir), expected_filename)

    # 반환된 파일 경로와 expected_path가 같은지 확인
    assert saved_file_path == expected_path, (
        f"파일명이 예상과 다릅니다. " f"예상: {expected_path}, 실제: {saved_file_path}"
    )

    print("테스트 성공: save_html 메서드가 Original URL 주석 삽입 및 올바른 파일명으로 저장을 정상적으로 수행했습니다.")


def test_integration_html_file_storage_and_extract_original_url(tmp_path):
    """
    HTMLFileStorage.save_html로 저장한 파일을
    extract_original_url로 다시 로드하여
    원본 URL이 올바르게 추출되는지를 검증하는 통합 테스트.
    """

    # 1) 임시 디렉토리에 저장 폴더 생성 및 스토리지 인스턴스 생성
    save_dir = tmp_path / "html_files"
    storage = HTMLFileStorage(str(save_dir))

    # 2) 테스트용 URL, HTML 콘텐츠
    test_url = "https://example.com/test/page?query=abc#section"
    test_html_content = "<html><body><h1>Hello World</h1></body></html>"

    # 3) HTML 저장
    saved_file_path = storage.save_html(test_url, test_html_content)

    # 4) 파일이 정상적으로 생성되었는지 확인
    assert os.path.exists(saved_file_path), "파일이 저장되지 않았습니다."

    # 5) extract_original_url 함수를 통해 원본 URL 추출
    extracted_url = extract_original_url(saved_file_path)

    # 6) 추출된 URL이 실제 URL과 일치하는지 검증
    assert extracted_url == test_url, (
        f"추출된 URL이 실제 URL과 일치하지 않습니다. " f"실제: {test_url}, 추출: {extracted_url}"
    )

    print(
        "통합 테스트 성공: HTMLFileStorage와 extract_original_url이 연동되어 원본 URL을 정상적으로 보존/추출합니다."
    )
