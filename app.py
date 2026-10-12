
from flask import Flask, render_template, request

from src.recommendation import MovieRecommendationSystem


app = Flask(__name__)


def load_recommender():
    """Load the saved movie recommendation artifacts."""
    try:
        recommender = MovieRecommendationSystem()
        recommender.load_artifacts()
        return recommender

    except FileNotFoundError as error:
        print(f"Required file error: {error}")
        return None

    except Exception as error:
        print(f"Error loading recommendation system: {error}")
        return None


recommender = load_recommender()


@app.route("/", methods=["GET", "POST"])
def home():
    try:
        recommendations = []
        movie_name = ""
        matched_title = None
        error_message = None

        if recommender is None:
            error_message = (
                "Recommendation system is not ready. "
                "Please train the model first."
            )

        if request.method == "POST" and recommender is not None:
            movie_name = request.form.get(
                "movie_name", ""
            ).strip()

            if not movie_name:
                error_message = "Please enter a movie title."

            else:
                matched_title, recommendations = (
                    recommender.recommend(
                        movie_name,
                        top_n=5,
                    )
                )

                if matched_title is None:
                    error_message = (
                        "Movie not found. Please try another title."
                    )

                elif not recommendations:
                    error_message = (
                        "No other movies are available to recommend."
                    )

        return render_template(
            "index.html",
            movie_name=movie_name,
            matched_title=matched_title,
            recommendations=recommendations,
            error_message=error_message,
        )

    except Exception as error:
        app.logger.exception(
            "An error occurred while processing the request."
        )

        return render_template(
            "index.html",
            movie_name="",
            matched_title=None,
            recommendations=[],
            error_message=(
                "Something went wrong while generating "
                "recommendations. Please try again."
            ),
        ), 500


if __name__ == "__main__":
    try:
        app.run(debug=True)

    except Exception as error:
        print(f"Flask application failed to start: {error}")
