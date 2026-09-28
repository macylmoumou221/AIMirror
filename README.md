# AIMirror – Suivi d'humeur assisté par IA 

Bienvenue dans AIMirror, notre toute première application d'IA réalisée en équipe pour apprendre à construire une solution de bout en bout. Elle surveille la webcam, identifie les émotions faciales avec DeepFace (ou FER en solution de repli) et raconte une journée d'humeur à travers un tableau de bord Streamlit.

## Fonctionnalités clés

- Capture vidéo en direct assistée par OpenCV et analyse d'émotions en temps réel.
- Score de confiance associé à l'émotion dominante détectée pour chaque image.
- Journalisation CSV avec horodatage : `data/mood_log.csv` se remplit automatiquement.
- Synthèse quotidienne : humeur dominante, nombre total de scans et confiance moyenne.
- Graphique de répartition généré chaque jour dans `visuals/daily_mood_chart.png`.
- Interface Streamlit interactive avec flux vidéo, contrôles de session et tableau des détections récentes.
- Minuterie facultative pour arrêter automatiquement une session après N minutes.

## Structure du projet

```
ai-mood-tracker/
├── app.py                 # Interface Streamlit et boucle temps réel
├── emotion_detector.py    # Analyse des émotions, journalisation et rapports
├── data/
│   └── mood_log.csv       # Journal CSV (créé au premier lancement)
├── visuals/
│   └── daily_mood_chart.png  # Graphique sauvegardé automatiquement
├── requirements.txt       # Dépendances Python
└── README.md
```

## Démarrage rapide (PowerShell Windows)

1. Créez un environnement virtuel :

	```powershell
	python -m venv ..\.venv
	..\.venv\Scripts\Activate.ps1
	```

2. Installez les dépendances :

	```powershell
	pip install -r requirements.txt
	```

	> 💡 Au premier lancement, DeepFace télécharge ses poids dans `C:\Users\<vous>\.deepface`. Pour une installation plus légère, retirez `deepface` du fichier des dépendances : l'application basculera automatiquement sur FER.

3. Démarrez le tableau de bord :

	```powershell
	python -m streamlit run app.py
	```

4. Dans le navigateur :
	- Choisissez l'index de webcam (0 pour la caméra intégrée).
	- Ajustez la durée d'arrêt automatique (0 = session illimitée).
	- Cliquez sur **Start session** et placez votre visage dans le cadre.
	- Suivez en direct les émotions détectées et les statistiques du jour.

## Données & visualisations

- Chaque détection est ajoutée à `data/mood_log.csv` avec l'émotion dominante, la confiance et la source.
- La colonne de droite affiche en continu les statistiques quotidiennes.
- `visuals/daily_mood_chart.png` est régénéré à chaque consultation de la section « Today's Summary ».
- Téléchargez l'historique complet via le bouton **Download mood log** ou affichez les 10 dernières entrées directement dans l'app.

## Notes techniques

- **Accès caméra :** assurez-vous qu'aucune autre application n'occupe la webcam.
- **Performance :** DeepFace donne des résultats fins mais demande des ressources. Ajustez le nombre d'images par seconde (1 à 10 FPS) pour trouver le bon équilibre.
- **Fuseau horaire :** les horodatages sont enregistrés en UTC pour faciliter les analyses multi-machines.

## Pistes d'amélioration

- Détection multi-visages pour analyser l'humeur d'un groupe.
- Rapports hebdomadaires et visualisation des tendances.
- Thèmes clair/sombre personnalisés dans Streamlit.
- Stockage dans SQLite pour un historique plus riche.
- Accélération GPU et modèles d'émotion affinés sur mesure.

## À propos du projet

- **Période :** février 2025 – octobre 2025
- **Rôle principal :** Macyl MOUMOU
- **Technologies :** Python, OpenCV, DeepFace, Streamlit, pandas, matplotlib
- **Objectif pédagogique :** première expérience complète en IA, de la capture en temps réel à la visualisation d'un tableau de bord.

---

# AIMirror – AI Mood Tracker 

AIMirror is our team’s very first end-to-end AI project. We built it as a learning journey to understand how real-time computer vision, lightweight analytics, and a Streamlit dashboard can come together to tell a story about someone’s mood.

## Highlights

- Live webcam capture powered by OpenCV with DeepFace as the default emotion engine (FER fallback included).
- Confidence scoring for every dominant emotion detection.
- Automatic logging to `data/mood_log.csv` with UTC timestamps and the capture source.
- Daily rollups that surface total scans, average confidence, and the day’s dominant feeling.
- A persistent bar chart in `visuals/daily_mood_chart.png` that updates whenever the dashboard loads.
- Streamlit interface with session controls, live feed, metrics, and a recent detections table.
- Optional session auto-stop to keep experimental runs short and focused.

## Project layout

```
ai-mood-tracker/
├── app.py                 # Streamlit entry point and real-time loop
├── emotion_detector.py    # Emotion inference, logging, and aggregation helpers
├── data/
│   └── mood_log.csv       # CSV log (auto-created on first run)
├── visuals/
│   └── daily_mood_chart.png  # Saved chart refreshed by the dashboard
├── requirements.txt       # Python dependencies
└── README.md
```

## Getting started (Windows PowerShell)

1. Create and activate a virtual environment one folder up (matching the project’s layout):

	```powershell
	python -m venv ..\.venv
	..\.venv\Scripts\Activate.ps1
	```

2. Install dependencies inside the activated environment:

	```powershell
	pip install -r requirements.txt
	```

	> 💡 On first launch DeepFace downloads weights to `C:\Users\<you>\.deepface`. Remove `deepface` from `requirements.txt` if you prefer a smaller footprint—the app automatically falls back to FER.

3. Launch the dashboard:

	```powershell
	python -m streamlit run app.py
	```

4. In the browser:
	- Pick your camera index (0 for most built-in webcams).
	- Set an auto-stop duration (0 keeps the session open).
	- Click **Start session** and stay in frame.
	- Watch the real-time overlay and daily metrics evolve together.

## Data flow & insights

- Each detection is appended to `data/mood_log.csv` with dominant emotion, confidence, and source.
- The right column of the dashboard summarizes the current day at a glance.
- `visuals/daily_mood_chart.png` is regenerated whenever the summary panel renders.
- Use the **Download mood log** button for the full CSV or inspect the latest 10 entries directly in-app.

## Configuration notes

- **Camera permissions:** close other apps that might hold onto the webcam.
- **Performance tuning:** DeepFace is accurate but heavier; adjust the FPS slider to balance speed versus precision.
- **Timestamps:** All logs are stored in UTC for consistent aggregation.

## What’s next

- Multi-face detection for group mood tracking.
- Weekly trend reports and rolling averages.
- Dark/light Streamlit theming.
- SQLite (or other) persistence for longer-term analytics.
- Optional GPU acceleration and fine-tuned emotion models.

## Project context

- **Stack:** Python, OpenCV, DeepFace, Streamlit, pandas, matplotlib
- **Why we built it:** To learn how to stitch together real-time CV, AI inference, data logging, and UI polish in a single app.
