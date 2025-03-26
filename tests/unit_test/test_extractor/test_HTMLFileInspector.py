import pytest
from docmesh.extractor.HTMLFileInspector import HTMLFileInspector


def test_read_html_with_original_url(tmp_path):
    """
    원본 URL 주석이 존재할 때,
    read_html가 해당 URL을 추출하고
    전체 HTML을 반환하는지 확인합니다.
    """
    # 1) 테스트용 HTML 작성
    original_url = "https://example.com/page?param=123"
    html_content = """<html>
<!-- Original URL: {0} -->
<head><title>Test</title></head>
<body><p>Hello World</p></body>
</html>""".format(
        original_url
    )

    # 2) 임시 파일 생성
    html_file = tmp_path / "test_with_url.html"
    html_file.write_text(html_content, encoding="utf-8")

    # 3) read_html 호출
    url_extracted, full_content = HTMLFileInspector.read_html(str(html_file))

    # 4) 결과 검증
    assert (
        url_extracted == original_url
    ), f"원본 URL이 {original_url}로 반환되어야 합니다. 실제: {url_extracted}"
    assert full_content == html_content, "파일의 전체 내용이 그대로 반환되어야 합니다."


def test_read_html_without_original_url(tmp_path):
    """
    원본 URL 주석이 없는 경우,
    read_html가 original_url=None을 반환하는지 확인합니다.
    """
    # 1) 원본 URL 주석이 없는 HTML
    html_content = """<html>
<head><title>Test</title></head>
<body><p>No URL comment here</p></body>
</html>
    """

    # 2) 임시 파일 생성
    html_file = tmp_path / "test_without_url.html"
    html_file.write_text(html_content, encoding="utf-8")

    # 3) read_html 호출
    url_extracted, full_content = HTMLFileInspector.read_html(str(html_file))

    # 4) 결과 검증
    assert url_extracted is None, "원본 URL 주석이 없으므로 original_url은 None이어야 합니다."
    assert full_content == html_content, "파일 전체 내용이 그대로 반환되어야 합니다."


def test_read_html_file_not_found():
    """
    존재하지 않는 파일을 읽으려 하면,
    FileNotFoundError가 발생하는지 확인합니다.
    """
    with pytest.raises(FileNotFoundError):
        HTMLFileInspector.read_html("non_existent_file.html")
