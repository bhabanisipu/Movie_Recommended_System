# Movie Recommendation System

A Flask web app that recommends movies based on a user-entered movie title by comparing genres using TF-IDF and cosine similarity.

## Features
- Search for a movie by title
- Finds exact or approximate matches
- Recommends similar movies based on genre similarity
- Displays similarity score and progress bar
- Simple Flask front-end with HTML/CSS

## Project Structure
- `app.py` - Flask app entry point
- `src/recommendation.py` - recommendation logic and model loading
- `src/train_model.py` - model training script
- `data/processed_movies.csv` - processed movie dataset
- `models/` - saved model artifacts
- `static/style.css` - styling
- `templates/index.html` - HTML template

## Requirements
Install dependencies from the project requirements file:

```bash
pip install -r requirements.txt
```

## Run the App
From the project root:

```bash
python app.py
```

## How It Works
1. The app loads saved movie data and trained model artifacts.
2. User enters a movie name.
3. The app finds the closest matching movie title.
4. It compares the selected movie against all others using cosine similarity.
5. It returns the top recommended movies.

## Notes
- If the model files are missing, run the training script first:

```bash
python -m src.train_model
```

- The project uses Flask, pandas, scikit-learn, and joblib.
