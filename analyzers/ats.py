from __future__ import annotations

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Quelques verbes d'action basiques (anglais/français)
_ACTION_VERBS = {
    'achieved','analyzed','built','collaborated','created','designed','developed','implemented','improved','increased',
    'reduced','optimized','managed','led','supported','maintained','tested','validated','deployed','automated','documented',
    'monitored','tracked','researched','integrated','réalisé','analysé','construit','collaboré','créé','conçu','développé',
    'mis en œuvre','amélioré','augmenté','réduit','optimisé','géré','dirigé','soutenu','maintenu','testé','validé',
    'déployé','automatisé','documenté','surveillé','suivi','recherché','intégré'
}

_SECTION_HEADERS = ["profile", "profile summary", "summary", "profil", "résumé", "resume", "objective" ,
                    "experience", "work experience", "expérience", "expériences", "professional experience", "employment history" ,
                    "education", "formation", "education & certifications", "academic qualifications", "studies", "éducation", "degrees" ,
                    "skills", "compétences", "technical skills", "skills & tools", "compétence", "abilities" ,
                    "languages", "langues", "language skills" ,
                    "projects", "academic projects", "projets", "personal projects", "portfolio" ,
                    "contact", "contactez", "informations personnelles", "personal info", "contact information" ,
                    "strengths & qualities", "strengths", "qualities", "personal qualities", "atouts"]


_CONTACT_KW = ['phone', 'email', 'linkedin', 'github', 'téléphone', 'courriel']


def calculate_ats_score(text: str) -> int:
    """Calcule un score ATS heuristique simple (0-100).

    Critères (pondération approximative) :
    - verbes d'action (0-30)
    - éléments quantifiables (0-25)
    - sections présentes (0-20)
    - longueur (0-15)
    - contact info (0-10)
    """
    if not text:
        return 0
    try:
        txt = (text or '').lower()

        words = re.findall(r"\w+", txt)
        word_count = len(words)

        # action verbs
        action_verbs_count = sum(1 for w in _ACTION_VERBS if w in txt)
        if action_verbs_count > 5:
            action_score = 30
        elif action_verbs_count > 2:
            action_score = 15
        else:
            action_score = 5 if action_verbs_count > 0 else 0

        # quantifiable patterns
        quant_patterns = [r'increased by \d+%', r'reduced by \d+%', r'\d+% improvement', r'saved \$?\d+', r'\d+\+', r'over \d+', r'by \d+']
        quant_count = sum(1 for p in quant_patterns if re.search(p, txt))
        if quant_count > 3:
            quant_score = 25
        elif quant_count > 1:
            quant_score = 10
        else:
            quant_score = 0

        # sections
        section_count = sum(1 for s in _SECTION_HEADERS if re.search(rf"\b{s}\b", txt))
        if section_count >= 4:
            section_score = 20
        elif section_count >= 2:
            section_score = 10
        else:
            section_score = 0

        # length
        if 600 <= word_count <= 800:
            length_score = 15
        elif 400 <= word_count <= 1000:
            length_score = 5
        else:
            length_score = 0

        # contact
        contact_count = sum(1 for k in _CONTACT_KW if re.search(rf"\b{k}\b", txt))
        contact_score = 10 if contact_count >= 2 else (5 if contact_count == 1 else 0)

        total = action_score + quant_score + section_score + length_score + contact_score
        return int(min(total, 100))
    except Exception as exc:
        logger.exception("Erreur compute ATS: %s", exc)
        return 0


def generate_recommendations(text: str, ats_score: int) -> List[str]:
    """Génère recommandations textuelles simples pour améliorer le CV.

    Retourne une liste de chaînes.
    """
    recs: List[str] = []
    txt = (text or '').lower()

    # action verbs
    action_verbs_present = sum(1 for v in _ACTION_VERBS if v in txt)
    if action_verbs_present < 3:
        recs.append("Ajoutez davantage de verbes d'action (ex: developed, implemented, managed) pour décrire vos réalisations.")

    # quantifiable
    if not re.search(r'\d', txt):
        recs.append("Ajoutez des résultats chiffrés lorsque possible (ex: 'Increased sales by 20%').")

    # sections
    missing = [s for s in ['experience', 'education', 'skills'] if not re.search(rf"\b{s}\b", txt)]
    if missing:
        recs.append(f"Ajoutez des en-têtes clairs pour les sections : {', '.join(missing)}.")

    # length
    word_count = len(re.findall(r"\w+", txt))
    if word_count < 400:
        recs.append("Votre CV est peut-être trop court. Ajoutez plus de détails sur vos expériences et compétences.")
    elif word_count > 1200:
        recs.append("Votre CV semble long. Essayez de le condenser (1-2 pages maximum).")

    # contact
    if not any(k in txt for k in _CONTACT_KW):
        recs.append("Incluez vos informations de contact (email, téléphone, LinkedIn).")

    # generic tips depending on score
    if ats_score < 50:
        recs.append("Considérez de restructurer votre CV avec des sections claires et des bullet points pour faciliter le parsing ATS.")
    if ats_score < 70:
        recs.append("Taillez et adaptez votre CV à chaque offre : incorporez des mots-clés importants de la description de poste.")

    return recs or ["Votre CV semble correct pour l'ATS. Pensez à le personnaliser pour chaque candidature."]

