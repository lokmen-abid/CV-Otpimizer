from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# Dossier pour sauvegarder temporairement les fichiers uploadés
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Charger les domaines disponibles (chemin robuste)
def _list_domains():
    domains_dir = os.path.join(app.root_path, 'config', 'domains')
    domains = []
    try:
        for fname in sorted(os.listdir(domains_dir)):
            if fname.endswith('.json'):
                domains.append(os.path.splitext(fname)[0])
    except Exception as exc:
        logger.debug('Impossible de lister les domaines: %s', exc)
    return domains


@app.route('/')
def index():
    domains = _list_domains()
    return render_template('index.html', domains=domains)


# Précharger le modèle embeddings en background si possible (itération 1 option A)
try:
    from analyzers.embeddings import preload_model  # type: ignore
    try:
        preload_model(background=True)
        logger.debug('Preload embeddings launched in background')
    except Exception as exc:
        logger.debug('Preload embeddings failed to start: %s', exc)
except Exception:
    # embeddings optional
    pass


# Route GET+POST pour traiter l'upload du CV et la JD
@app.route('/improve', methods=['GET', 'POST'])
def improve():
    if request.method == 'POST':
        # Déléguer la logique de traitement à services.service.handle_improve
        try:
            from services.service import handle_improve  # type: ignore
        except Exception as exc:
            logger.exception('Impossible d\'importe le service de traitement: %s', exc)
            return render_template('improve.html', cv_filename=None, job_description=None, cv_sections=None, cv_lang=None, cv_text=None, jd_lang=None)

        cv_file = request.files.get('cv_file')
        jd_text = request.form.get('job_description', '').strip()
        domain = request.form.get('domain') or None
        cv_lang_choice = request.form.get('cv_lang') or None
        jd_lang_choice = request.form.get('jd_lang') or None

        ctx = handle_improve(cv_file, jd_text, domain, cv_lang_choice, jd_lang_choice, app.config['UPLOAD_FOLDER'])

        return render_template('improve.html', **ctx)

    # GET: afficher la page (vide ou default)
    return render_template('improve.html',
                           cv_filename=None,
                           job_description=None,
                           cv_sections=None,
                           cv_lang=None,
                           cv_text=None,
                           jd_lang=None)


# route /upload supprimée (non utilisée) - code nettoyé

if __name__ == '__main__':
    # démarrer en mode debug pour voir les logs
    port = int(os.environ.get('PORT', 5000))  # Render fournit PORT
    app.run(host='0.0.0.0', port=port, debug=False)
