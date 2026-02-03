"""
services/service.py

Contient la logique centralisée pour le traitement d'une soumission "improve":
- sauvegarde du CV uploadé
- extraction / parsing via parsers
- calcul ATS (analyzers.ats)
- calcul similarités (TF-IDF, embeddings) si disponibles
- génération des recommandations

La fonction publique principale est `handle_improve` qui renvoie un dictionnaire prêt à être passé
à `render_template('improve.html', **ctx)` dans `app.py`.

Le module est défensif : toutes les importations optionnelles sont protégées par try/except
et les chemins où une dépendance manque renvoient des valeurs par défaut.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)


def _save_uploaded_file(file_obj, upload_folder: str) -> Optional[str]:
    """Sauvegarde le fichier uploadé dans `upload_folder` et renvoie le nom de fichier sauvegardé.

    Retourne None si aucun fichier ou erreur.
    """
    if not file_obj or not getattr(file_obj, 'filename', None):
        return None
    try:
        filename = secure_filename(file_obj.filename)
        os.makedirs(upload_folder, exist_ok=True)
        path = os.path.join(upload_folder, filename)
        file_obj.save(path)
        return filename
    except Exception as exc:
        logger.exception("Erreur sauvegarde fichier upload: %s", exc)
        return None


def handle_improve(cv_file, jd_text: str, domain: Optional[str], cv_lang_choice: Optional[str], jd_lang_choice: Optional[str], upload_folder: str) -> Dict[str, Any]:
    """Orchestre le traitement du CV et de la Job Description.

    Args:
        cv_file: FileStorage ou None
        jd_text: texte de la JD (string)
        domain: domaine choisi par l'utilisateur (nom du fichier JSON sans extension) ou None
        cv_lang_choice: choix de langue CV (e.g., 'fr','en','auto')
        jd_lang_choice: choix de langue JD
        upload_folder: dossier où sauvegarder les uploads

    Returns:
        ctx: dict utilisable comme kwargs pour render_template('improve.html', **ctx)
    """
    # Valeurs par défaut retournées
    ctx: Dict[str, Any] = {
        'cv_filename': None,
        'job_description': jd_text,
        'cv_sections': None,
        'cv_lang': None,
        'cv_text': '',
        'jd_lang': None,
        'ats_score': 0,
        'ats_res': {'total': 0, 'breakdown': {}},
        'recommendations': [],
        'match_results': {'similarity_score': 0.0, 'tfidf_method': 'unavailable', 'jd_top_terms': [], 'cv_top_terms': [], 'keyword_match': 0.0, 'matching_keywords': [], 'embed_score': 0.0, 'embed_method': 'unavailable'},
        'keywords_table': [],
        'error': None,
    }

    # Importer modules optionnels de manière défensive
    try:
        from parsers.cv_parser import parse_cv, clean_text, detect_sections, detect_language  # type: ignore
    except Exception:
        parse_cv = clean_text = detect_sections = detect_language = None

    try:
        from parsers.jd_parser import parse_jd  # type: ignore
    except Exception:
        parse_jd = None

    try:
        from analyzers.ats import calculate_ats_score, generate_recommendations  # type: ignore
    except Exception:
        calculate_ats_score = generate_recommendations = None

    try:
        from analyzers.similarity_scorer import compute_tfidf_similarity, keyword_overlap, compute_embedding_similarity  # type: ignore
    except Exception:
        compute_tfidf_similarity = keyword_overlap = compute_embedding_similarity = None

    # 1) Sauvegarder le fichier uploadé
    cv_filename = _save_uploaded_file(cv_file, upload_folder)
    ctx['cv_filename'] = cv_filename

    cv_text = ''
    cv_sections = None
    cv_lang = None

    # 2) Extraire et parser le CV
    if cv_filename:
        save_path = os.path.join(upload_folder, cv_filename)
        ext = os.path.splitext(cv_filename)[1].lower()
        try:
            if parse_cv is not None and ext in ('.pdf', '.docx'):
                parsed = parse_cv(save_path)
                cv_text = parsed.get('cleaned_text', '') or parsed.get('raw_text', '')
                cv_lang = parsed.get('language')
                cv_sections = parsed.get('sections')
            else:
                # fallback lecture simple
                if ext in ('.txt', '.md', '.doc') or parse_cv is None:
                    with open(save_path, 'r', encoding='utf-8', errors='replace') as fh:
                        raw = fh.read()
                    cv_text = clean_text(raw) if clean_text is not None else raw
                    cv_lang = detect_language(cv_text) if detect_language is not None else None
                    cv_sections = detect_sections(cv_text, lang=cv_lang) if detect_sections is not None else None
        except Exception as exc:
            logger.exception('Erreur pendant l\'analyse du CV: %s', exc)
            ctx['error'] = f"Erreur pendant l'analyse du CV: {exc}"
            # continuer, cv_text pourra être vide

    # 3) Traiter la Job Description
    jd_parsed = {'text': jd_text, 'lang': None}
    if jd_text and parse_jd is not None:
        try:
            jd_parsed = parse_jd(jd_text, from_file=False)
        except Exception:
            jd_parsed = {'text': jd_text, 'lang': None}

    # 4) override languages si l'utilisateur a précisé
    if cv_lang_choice and cv_lang_choice != 'auto':
        cv_lang = cv_lang_choice
    if jd_lang_choice and jd_lang_choice != 'auto':
        jd_parsed['lang'] = jd_lang_choice

    ctx['cv_text'] = cv_text
    ctx['cv_sections'] = cv_sections
    ctx['cv_lang'] = cv_lang
    ctx['job_description'] = jd_parsed.get('text')
    ctx['jd_lang'] = jd_parsed.get('lang')

    # Si pas de CV, retourner rapidement (le front-end gère l'affichage)
    if not cv_text:
        return ctx

    # 5) Calculer ATS
    try:
        if calculate_ats_score is not None:
            ats_res = calculate_ats_score(cv_text, domain=domain, lang=cv_lang)
            ctx['ats_res'] = ats_res
            ctx['ats_score'] = ats_res.get('total', 0)
        else:
            ctx['ats_res'] = {'total': 0, 'breakdown': {}}
            ctx['ats_score'] = 0
    except Exception as exc:
        logger.debug('Erreur calculate_ats_score: %s', exc)
        ctx['ats_res'] = {'total': 0, 'breakdown': {}}
        ctx['ats_score'] = 0

    # 6) Similarité TF-IDF
    match_results = ctx['match_results']
    try:
        if compute_tfidf_similarity is not None and jd_parsed.get('text'):
            tfidf_res = compute_tfidf_similarity(cv_text, jd_parsed.get('text'), lang=jd_parsed.get('lang') or cv_lang, top_n=20)
            match_results.update({
                'similarity_score': round((tfidf_res.get('score') or 0.0) * 100.0, 1),
                'tfidf_method': tfidf_res.get('tfidf_method'),
                'jd_top_terms': tfidf_res.get('jd_top_terms', []),
                'cv_top_terms': tfidf_res.get('cv_top_terms', [])
            })
        else:
            match_results.update({'similarity_score': 0.0, 'tfidf_method': 'unavailable', 'jd_top_terms': [], 'cv_top_terms': []})
    except Exception as exc:
        logger.debug('Erreur compute_tfidf_similarity: %s', exc)
        match_results.update({'similarity_score': 0.0, 'tfidf_method': 'error', 'jd_top_terms': [], 'cv_top_terms': []})

    # 7) Keyword overlap
    try:
        if keyword_overlap is not None and jd_parsed.get('text'):
            kw = keyword_overlap(cv_text, jd_parsed.get('text'))
            match_results['keyword_match'] = kw.get('keyword_match')
            match_results['matching_keywords'] = kw.get('matching_keywords')
        else:
            match_results['keyword_match'] = 0.0
            match_results['matching_keywords'] = []
    except Exception as exc:
        logger.debug('Erreur keyword_overlap: %s', exc)
        match_results['keyword_match'] = 0.0
        match_results['matching_keywords'] = []

    # 8) Embeddings similarity (optionnel)
    try:
        if compute_embedding_similarity is not None and jd_parsed.get('text'):
            emb = compute_embedding_similarity(cv_text, jd_parsed.get('text'))
            if isinstance(emb, dict):
                match_results['embed_score'] = round((emb.get('score') or 0.0) * 100.0, 1)
                match_results['embed_method'] = emb.get('method')
            else:
                match_results['embed_score'] = 0.0
                match_results['embed_method'] = 'unavailable'
    except Exception as exc:
        logger.debug('Erreur compute_embedding_similarity: %s', exc)
        match_results['embed_score'] = 0.0
        match_results['embed_method'] = 'error'

    ctx['match_results'] = match_results

    # 9) Recommandations
    try:
        if generate_recommendations is not None:
            recommendations = generate_recommendations(cv_text, ctx.get('ats_res', {}), domain=domain, lang=cv_lang, jd_text=jd_parsed.get('text'), match_results=match_results)
            ctx['recommendations'] = recommendations
        else:
            ctx['recommendations'] = []
    except Exception as exc:
        logger.debug('Erreur generate_recommendations: %s', exc)
        ctx['recommendations'] = []

    # 10) keywords table
    ctx['keywords_table'] = match_results.get('matching_keywords', [])

    return ctx
