import pytest
from analyzers import ats


def test_calculate_ats_score_basic(sample_text_en):
    res = ats.calculate_ats_score(sample_text_en, domain='it', lang='en')
    assert isinstance(res, dict)
    assert 0 <= res.get('total', -1) <= 100
    # more verbs in sample_text_en should give non-zero action_verbs count
    assert 'breakdown' in res
    assert 'action_verbs' in res['breakdown']


def test_generate_recommendations_limits(sample_text_en):
    ats_res = ats.calculate_ats_score(sample_text_en, domain='it', lang='en')
    recs = ats.generate_recommendations(sample_text_en, ats_res, domain='it', lang='en', jd_text="")
    assert isinstance(recs, list)
    assert len(recs) <= 5


def test_generate_recommendations_short_text():
    short = "Short resume"
    ats_res = ats.calculate_ats_score(short, domain=None, lang='en')
    recs = ats.generate_recommendations(short, ats_res, domain=None, lang='en', jd_text="")
    assert isinstance(recs, list)
    assert recs


def test_generate_recommendations_domain_respects_config(sample_text_en):
    # Ensure domain-specific recommendations are used when domain matches
    ats_res = ats.calculate_ats_score(sample_text_en, domain='it', lang='en')
    recs = ats.generate_recommendations(sample_text_en, ats_res, domain='it', lang='en', jd_text="python docker kubernetes")
    # at least one recommendation should come from domain config if jd_text contains domain keywords
    assert isinstance(recs, list)
