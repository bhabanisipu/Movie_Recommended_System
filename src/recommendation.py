
from pathlib import Path
import difflib

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed_movies.csv"
)
VECTORIZER_PATH = (
    PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
)
FEATURES_PATH = (
    PROJECT_ROOT / "models" / "movie_features.pkl"
)


class MovieRecommendationSystem:

    def __init__(self):
        self.movies = None
        self.vectorizer = None
        self.movie_features = None
        self.movie_titles = []

    def load_artifacts(self):
        """Load processed data and saved model artifacts."""
        try:
            required_paths = [
                PROCESSED_DATA_PATH,
                VECTORIZER_PATH,
                FEATURES_PATH,
            ]

            for path in required_paths:
                if not path.exists():
                    raise FileNotFoundError(
                        f"Required file not found: {path}\n"
                        "Run training first using "
                        "'python -m src.train_model'."
                    )

            self.movies = pd.read_csv(PROCESSED_DATA_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.movie_features = joblib.load(FEATURES_PATH)

            if self.movies.empty:
                raise ValueError("Processed movie dataset is empty.")

            if self.movie_features.shape[0] != len(self.movies):
                raise ValueError(
                    "Movie data and feature matrix are misaligned. "
                    "Please retrain the model."
                )

            self.movie_titles = self.movies["title"].tolist()

            print("Saved artifacts loaded successfully.")
            return self

        except FileNotFoundError as error:
            print(f"File error: {error}")
            raise

        except (ValueError, OSError) as error:
            print(f"Artifact loading error: {error}")
            raise

        except Exception as error:
            print(f"Unexpected loading error: {error}")
            raise

    def find_movie(self, movie_name):
        """Find an exact or approximate movie title."""
        try:
            if self.movies is None:
                raise RuntimeError(
                    "Call load_artifacts() before searching."
                )

            if not isinstance(movie_name, str):
                raise TypeError("Movie name must be a string.")

            movie_name = movie_name.strip()

            if not movie_name:
                return None

            # Case-insensitive exact match
            matching_movies = self.movies[
                self.movies["title"].str.casefold()
                == movie_name.casefold()
            ]

            if not matching_movies.empty:
                return int(matching_movies.index[0])

            # Approximate match for spelling differences
            matches = difflib.get_close_matches(
                movie_name,
                self.movie_titles,
                n=1,
                cutoff=0.4,
            )

            if not matches:
                return None

            matched_title = matches[0]

            matching_indices = self.movies.index[
                self.movies["title"] == matched_title
            ]

            if len(matching_indices) == 0:
                return None

            return int(matching_indices[0])

        except (TypeError, RuntimeError) as error:
            print(f"Movie search error: {error}")
            raise

        except Exception as error:
            print(f"Unexpected search error: {error}")
            raise

    def recommend(self, movie_name, top_n=5):
        """Return the most similar movies based on genre."""
        try:
            if self.movie_features is None:
                raise RuntimeError(
                    "Call load_artifacts() before recommending."
                )

            if not isinstance(top_n, int) or isinstance(top_n, bool):
                raise TypeError("top_n must be an integer.")

            if top_n < 1:
                raise ValueError("top_n must be at least 1.")

            movie_index = self.find_movie(movie_name)

            if movie_index is None:
                return None, []

            similarity_scores = cosine_similarity(
                self.movie_features[movie_index],
                self.movie_features,
            ).flatten()

            sorted_indices = similarity_scores.argsort()[::-1]

            recommendations = []

            for index in sorted_indices:
                if index == movie_index:
                    continue

                movie = self.movies.iloc[index]

                recommendations.append({
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "similarity": round(
                        float(similarity_scores[index]), 4
                    ),
                })

                if len(recommendations) >= top_n:
                    break

            matched_title = self.movies.iloc[movie_index]["title"]

            return matched_title, recommendations

        except (TypeError, ValueError, RuntimeError) as error:
            print(f"Recommendation error: {error}")
            raise

        except Exception as error:
            print(f"Unexpected recommendation error: {error}")
            raise


if __name__ == "__main__":
    try:
        recommender = MovieRecommendationSystem()
        recommender.load_artifacts()

        movie_name = input(
            "Enter your favourite movie name: "
        )

        matched_title, recommendations = recommender.recommend(
            movie_name,
            top_n=5,
        )

        if matched_title is None:
            print(
                "No matching movie found. "
                "Please try another title."
            )
        else:
            print(f"\nRecommendations for: {matched_title}\n")

            for rank, movie in enumerate(
                recommendations, start=1
            ):
                print(f"{rank}. {movie['title']}")
                print(f"   Genres: {movie['genres']}")
                print(
                    f"   Similarity: {movie['similarity']:.4f}"
                )
                print("-" * 40)

    except FileNotFoundError:
        print(
            "Required files are missing. "
            "Train the model before running recommendations."
        )

    except (ValueError, TypeError, RuntimeError) as error:
        print(f"Could not generate recommendations: {error}")

    except Exception as error:
        print(f"Unexpected application error: {error}")
