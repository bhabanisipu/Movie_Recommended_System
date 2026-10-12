
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
FEATURES_PATH = MODEL_DIR / "movie_features.pkl"


def train_model():
    """Train the TF-IDF representation of movie genres."""
    try:
        movies = preprocess_data()
        vectorizer = TfidfVectorizer()
        movie_features = vectorizer.fit_transform(
            movies["genres"]
        )

        if movie_features.shape[0] != len(movies):
            raise ValueError(
                "Feature count does not match movie count."
            )
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(movie_features, FEATURES_PATH)

        print("\nTF-IDF training completed successfully.")
        print("Number of movies:", len(movies))
        print("Feature matrix shape:", movie_features.shape)
        print("Vectorizer saved to:", VECTORIZER_PATH)
        print("Movie features saved to:", FEATURES_PATH)

        return vectorizer, movie_features

    except (ValueError, OSError) as error:
        print(f"Training or file error: {error}")
        raise

    except Exception as error:
        print(f"Unexpected training error: {error}")
        raise


if __name__ == "__main__":
    try:
        train_model()

    except Exception:
        print("Model training failed. Check the error messages above.")
