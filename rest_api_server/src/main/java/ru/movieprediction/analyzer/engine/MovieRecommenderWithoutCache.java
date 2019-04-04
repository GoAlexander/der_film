package ru.movieprediction.analyzer.engine;

import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import ru.movieprediction.analyzer.model.RecommendationDTO;

import java.time.LocalDateTime;

@Service
@ConditionalOnProperty(value = "predictions.caching.enabled", havingValue = "false")
public class MovieRecommenderWithoutCache implements MovieRecommender {

    @Value("path-to-analyzer")
    String pathToAnalyzer;

    @Override
    public RecommendationDTO recommend(PredictionScript predictionScript, Integer userId, Integer amount) {
        RecommendationDTO recommendationDTO = new RecommendationDTO();
        recommendationDTO.setUserId(userId);
        recommendationDTO.setNumOfMovies(amount);
        String recommendations = Executor.executePredictor(predictionScript, userId.toString(), amount.toString());
        JSONParser parser = new JSONParser();
        JSONArray jsonArray = null;
        try {
            jsonArray = (JSONArray) parser.parse(recommendations);
        } catch (ParseException e) {
            e.printStackTrace();
        }
        recommendationDTO.setMovies(jsonArray);
        recommendationDTO.setRecommendTime(LocalDateTime.now());
        return recommendationDTO;
    }
}
