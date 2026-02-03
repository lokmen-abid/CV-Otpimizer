import pytest
from parsers import cv_parser, jd_parser


def test_clean_text_normalization():
    raw = "Hello\r\nWorld\rTest\n\n\nExtra"
    cleaned = cv_parser.clean_text(raw)
    assert "\r" not in cleaned
    assert cleaned.count("\n\n") <= 1


def test_detect_language_en_and_fr(sample_text_en, sample_text_fr):
    assert cv_parser.detect_language(sample_text_en) == "en"
    assert cv_parser.detect_language(sample_text_fr) == "fr"


def test_split_sections_by_headers_with_headers():
    txt = "PROFILE:\nThis is profile.\n\nEXPERIENCE:\nJob details here.\n\nEDUCATION:\nSchool"
    sections = cv_parser.split_sections_by_headers(txt)
    assert "profile" in sections
    assert "experience" in sections
    assert "education" in sections
    assert "This is profile." in sections["profile"]


def test_extract_text_docx_file(sample_docx_path):
    text = cv_parser.extract_text(sample_docx_path)
    assert text


def test_extract_text_txt_file(sample_txt_path):
    text = cv_parser.extract_text(sample_txt_path)
    assert "Experienced software" in text or text != ""


def test_parse_cv_integration(sample_txt_path):
    res = cv_parser.parse_cv(sample_txt_path, filename=sample_txt_path)
    assert "raw_text" in res and "cleaned_text" in res and "language" in res and "sections" in res
    assert res["language"] in ("en", "fr")


def test_jd_clean_and_detect_language():
    txt_en = "We are looking for a Data Scientist with Python and SQL skills."
    txt_fr = "Nous recherchons un Data Scientist avec Python et SQL."
    cleaned_en = jd_parser.clean_jd_text(txt_en)
    cleaned_fr = jd_parser.clean_jd_text(txt_fr)
    assert cleaned_en
    assert cleaned_fr
    assert jd_parser.detect_language_jd(cleaned_fr) == "fr"
    assert jd_parser.detect_language_jd(cleaned_en) == "en"


def test_read_text_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        jd_parser.read_text_file(str(tmp_path / "nofile.txt"))
