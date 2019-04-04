package ru.movieprediction.analyzer.engine;

import ru.movieprediction.analyzer.model.RecommendationDTO;

public interface MovieRecommender {
    RecommendationDTO recommend(PredictionScript predictionScript, Integer userId, Integer amount);

    default void clearCache() {}
}
