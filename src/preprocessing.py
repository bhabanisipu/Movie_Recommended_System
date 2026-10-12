import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "movies.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed_movies.csv"

def load_movie_data():
    """Load the original movie dataset."""
    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {DATA_PATH}"
            )

        movies = pd.read_csv(DATA_PATH)

        if movies.empty:
            raise ValueError("The original dataset is empty.")

        required_columns = {"title", "genres"}
        missing_columns = required_columns - set(movies.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        print("Original dataset loaded successfully.")
        return movies

    except FileNotFoundError as error:
        print(f"File error: {error}")
        raise

    except pd.errors.EmptyDataError as error:
        print(f"CSV error: The file has no data. {error}")
        raise

    except pd.errors.ParserError as error:
        print(f"CSV parsing error: {error}")
        raise

    except PermissionError as error:
        print(f"Permission error: {error}")
        raise

    except Exception as error:
        print(f"Unexpected error while loading data: {error}")
        raise


def clean_movie_data(movies):
    """Clean and validate the movie dataset."""
    try:
        if movies is None or movies.empty:
            raise ValueError("No movie data was provided.")

        movies = movies.copy()

        # Remove rows with missing title or genre
        movies = movies.dropna(subset=["title", "genres"])

        # Clean text values
        movies["title"] = movies["title"].astype(str).str.strip()
        movies["genres"] = movies["genres"].astype(str).str.strip()

        # Remove empty title or genre values
        movies = movies[
            (movies["title"] != "")
            & (movies["genres"] != "")
        ]

        # Remove duplicate movie records
        movies = movies.drop_duplicates(
            subset=["title", "genres"]
        )

        # Reset index so saved features align with movie rows
        movies = movies.reset_index(drop=True)

        if movies.empty:
            raise ValueError(
                "No valid movie records remain after cleaning."
            )

        print("Movie data cleaned successfully.")
        print("Processed dataset shape:", movies.shape)

        return movies

    except (KeyError, ValueError) as error:
        print(f"Data cleaning error: {error}")
        raise

    except Exception as error:
        print(f"Unexpected cleaning error: {error}")
        raise


def save_processed_data(movies):
    """Save the cleaned dataset inside the data folder."""
    try:
        PROCESSED_DATA_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        movies.to_csv(PROCESSED_DATA_PATH, index=False)

        print(f"Processed data saved to: {PROCESSED_DATA_PATH}")

    except PermissionError as error:
        print(f"Permission error while saving processed data: {error}")
        raise

    except OSError as error:
        print(f"File system error while saving processed data: {error}")
        raise

    except Exception as error:
        print(f"Unexpected error while saving processed data: {error}")
        raise


def preprocess_data():
    """Run the complete preprocessing workflow."""
    try:
        movies = load_movie_data()
        processed_movies = clean_movie_data(movies)
        save_processed_data(processed_movies)

        return processed_movies

    except Exception as error:
        print(f"Preprocessing workflow failed: {error}")
        raise


if __name__ == "__main__":
    try:
        processed_movies = preprocess_data()

        print("\nPreprocessing completed successfully.")
        print(processed_movies[["title", "genres"]].head())

    except Exception:
        print("Preprocessing could not be completed.")