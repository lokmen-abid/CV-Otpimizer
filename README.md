# ATS CV-Optimizer

Présentation du projet
----------------------
ATS CV-Optimizer est une application Python/Flask conçue pour analyser un CV (résumé) par rapport à une Job Description (JD) et calculer un score d'adéquation pour les systèmes de recrutement automatisés (ATS). L'objectif principal est de fournir :
- une extraction robuste du texte depuis des CV (PDF, DOCX, TXT),
- une détection de la langue et une segmentation en sections (expérience, éducation, compétences, ...),
- un calcul heuristique d'un score ATS et des recommandations simples pour améliorer le CV,
- une comparaison CV ↔ JD par similarité (TF‑IDF et embeddings dans l'itération suivante).

Ce dépôt implémente l'itération 0 (fonctionnalités de base) et prépare l'itération 1 (embeddings et améliorations via IA/OpenAI).

Usage rapide
------------
- Démarrer l'application :
  - Créez un environnement virtuel Python (recommandé).
  - Installez les dépendances : `pip install -r requirements.txt` (voir remarques plus bas).
  - Lancer : `python app.py` puis ouvrir `http://127.0.0.1:5000/`.
- Sur la page d'accueil, téléversez un CV (PDF/DOCX/TXT) et collez la Job Description, puis cliquez sur "Améliorer ton CV".

Remarques : plusieurs fonctionnalités avancées (KeyBERT, sentence-transformers, modèles spaCy) sont optionnelles et listées dans `requirements.txt`. Si elles ne sont pas installées, le code utilise des fallback ou retourne des résultats partiels.

Technologies et bibliothèques (présentes actuellement)
------------------------------------------------------
- Flask : micro-framework web pour l'interface (routes, templates).
- python-docx : extraction texte depuis fichiers .docx.
- PyMuPDF (fitz) : extraction texte depuis fichiers PDF.
- spaCy : NLP (tokenisation, segmentation, optionnel pour améliorer la détection de sections).
- sentence-transformers : embeddings (prévu pour l'itération 1; optionnel).
- scikit-learn / numpy : TF‑IDF et calculs de similarité (optionnel selon installation).

Rôle de chaque composant
------------------------
- `app.py` : point d'entrée Flask. Routes principales :
  - `/` : page d'accueil (`templates/index.html`) contenant le formulaire d'upload.
  - `/improve` : route GET/POST qui gère l'upload du CV, la lecture de la JD, appelle les parsers et analyzers, puis rend `templates/improve.html`.
  L'application est defensive : elle importe les modules (parsers/analyzers) avec `try/except` et active des fallback si une dépendance est manquante.

- `parsers/cv_parser.py` : extraction et traitement du CV
  - extract_text(file, filename=None) -> str
    - détecte le type (pdf/docx/text) et retourne le texte brut.
    - accepte chemin, bytes ou file-like.
  - extract_text_from_pdf(file) -> str
  - extract_text_from_docx(file) -> str
  - clean_text(text, collapse_newlines=True) -> str
    - normalisation Unicode, suppression de caractères non imprimables, réduction des sauts de ligne.
  - detect_language(text) -> 'fr'|'en'
    - heuristique simple basée sur mots indicateurs.
  - split_sections_by_headers / detect_sections(text, lang=None) -> dict
    - heuristiques regex + spaCy (si installé) pour repérer des titres de sections et retourner un mapping `section_name -> contenu`.
  - post_process_sections(sections) -> dict
    - nettoyages spécifiques (fusion de petits bloc, déplacement de langues, ...).
  - parse_cv(file, filename=None, do_clean=True) -> dict
    - orchestration : extrait raw_text, cleaned_text, language et sections.

- `parsers/jd_parser.py` : traitement des Job Descriptions
  - read_text_file(path) -> str
  - clean_jd_text(text, collapse_newlines=True) -> str
  - detect_language_jd(text) -> 'fr'|'en'
  - parse_jd(source, from_file=False, do_clean=True) -> dict

- `analyzers/ats.py` : calcul heuristique du score ATS
  - calculate_ats_score(text) -> int (0-100)
    - critères : verbes d'action, éléments quantifiables, sections présentes, longueur, contact.
  - generate_recommendations(text, ats_score) -> List[str]
    - recommandations textuelles basées sur les mêmes heuristiques.

- `analyzers/similarity_scorer.py` : similarité CV ↔ JD (TF‑IDF + embeddings)
  - compute_tfidf_similarity(cv_text, jd_text, lang=None, top_n=20) -> dict | None
    - utilise `sklearn` si disponible, sinon fallback simple. Retourne score (cosine) et top terms.
  - keyword_overlap(cv_text, jd_text) -> dict
    - overlap token set simple et pourcentage.
  - compute_embedding_similarity(cv_text, jd_text, model_name='all-MiniLM-L6-v2') -> dict | None
    - tente `sentence-transformers`, ensuite vecteurs spaCy, puis fallback Jaccard.

- `analyzers/keywords_extractor.py` : (désactivé)
  - actuellement remplacé par un stub vide pour garder les imports existants. Si vous souhaitez réactiver l'extraction de mots-clés (KeyBERT / YAKE / spaCy), remplacer ce fichier par les fonctions désirées.

- `templates/index.html` et `templates/improve.html` : interface utilisateur minimale (Bootstrap) pour l'upload et l'affichage des résultats.

Structure globale / architecture
-------------------------------
- `app.py` (Flask) orchestrateur → appelle
  - `parsers/` (extraction + nettoyage + segmentation)
  - `analyzers/` (calcul ATS, similarité)
- `templates/` pour la vue
- `uploads/` dossier local pour stocker temporairement les CV uploadés


Installation & dépendances
---------------------------
Le fichier `requirements.txt` liste les dépendances recommandées. Points importants :
- Pour une installation minimale (itération 0) : `Flask`, `python-docx`, `PyMuPDF`, `numpy`, `scikit-learn` (si vous voulez TF‑IDF avancé), `spaCy` (optionnel).
- Pour l'itération 1 (embeddings) : `sentence-transformers` et ses dépendances (PyTorch). Le téléchargement de modèles peut se faire au runtime et nécessite accès réseau.
- KeyBERT / YAKE / transformers sont listés comme optionnels pour extraction de mots-clés.

Exemples de commandes :

- Créer un venv et installer :

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

- Lancer l'app en développement :

```powershell
python app.py
```

Prochaines étapes (roadmap courte)
----------------------------------
1. Itération 1 : intégrer `sentence-transformers` pour embeddings et améliorer la page de résultats (ajouter heatmap / mise en évidence des mots manquants).
2. Intégration IA (OpenAI/LiteLLM) pour générer suggestions de réécriture et phrases d'accroche adaptées à la JD.
3. Ajouter des tests unitaires pour `parsers` et `analyzers` et un petit jeu d'exemples JD/CV pour CI.
4. Améliorer l'UX : télécharger un CV modifié (suggestions appliquées) et historique des analyses.


Ressources et références
------------------------
Voici quelques articles et notebooks utiles qui ont inspiré ou peuvent aider pour l'implémentation des scores ATS et des extractions :

- Notebook Kaggle (exemple d'implémentation ATS / scoring) :
  https://www.kaggle.com/code/moonmughal786/ats-based-resume-cv-score-checker

- Article (explication de comment fonctionne le scoring CV / ATS) :
  https://prompt-inspiration.com/blog/developpement/comment-fonctionne-scoring-cv-ats


Contact / notes
----------------
- Fichiers principaux à consulter : `app.py`, `parsers/cv_parser.py`, `parsers/jd_parser.py`, `analyzers/ats.py`, `analyzers/similarity_scorer.py`.
- Le code est volontairement défensif : il essaie d'importer des composants optionnels et utilise des fallback si nécessaire. Le but actuel est de minimiser les erreurs/warnings en évitant d'importer des modules non utilisés et en gardant des stubs clairs.
- Si vous souhaitez que j'implémente l'itération 1 (embeddings) ou la génération IA, indiquez-le et je préparerai les fichiers et tests associés.

Licence
-------
Projet personnel / prototype.
