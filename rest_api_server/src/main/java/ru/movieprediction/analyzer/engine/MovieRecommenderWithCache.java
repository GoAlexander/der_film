package ru.movieprediction.analyzer.engine;

import org.ehcache.Cache;
import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import ru.movieprediction.analyzer.model.RecommendationDTO;

import java.time.LocalDateTime;

@Service
@ConditionalOnProperty(value = "predictions.caching.enabled", havingValue = "true", matchIfMissing = true)
public class MovieRecommenderWithCache implements MovieRecommender {

    private static Cache<Integer, RecommendationDTO> cache;

    @Autowired
    public MovieRecommenderWithCache(Cache<Integer, RecommendationDTO> cache) {
        MovieRecommenderWithCache.cache = cache;
    }

    public RecommendationDTO recommend(PredictionScript predictionScript, Integer userId, Integer amount) {
        //TODO Resolve python execution error on windows
        // Discuss return format of python script

        if(cache.containsKey(userId)) {
            return cache.get(userId);
        }

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
        cache.put(userId, recommendationDTO);
        return recommendationDTO;
    }

    @Override
    public void clearCache() {
        cache.clear();
    }
}
