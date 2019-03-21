package ru.movieprediction.analyzer;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;
import ru.movieprediction.analyzer.engine.MovieRecommender;
import ru.movieprediction.analyzer.engine.PredictionScript;
import ru.movieprediction.analyzer.model.RecommendationDTO;

@RestController
public class ApiController {

    private final MovieRecommender recommender;

    @Autowired
    public ApiController(MovieRecommender recommender) {
        this.recommender = recommender;
    }

    @GetMapping(value="/get-films/{userId}/{amount}", produces=MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<RecommendationDTO> getRecommendations(@PathVariable String userId, @PathVariable String amount) {
        //TODO Cache results for users (?)
        //TODO Discuss return format
        Integer id;
        Integer amountOfMovies;

        try {
            id = Integer.parseInt(userId);
            amountOfMovies = Integer.parseInt(amount);
        }
        catch (NumberFormatException ex) {
            //TODO Better handling or refactoring
            ex.printStackTrace();
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(null);
        }
        return ResponseEntity.ok(recommender.recommend(PredictionScript.FAVORITES, id, amountOfMovies));
    }

    @GetMapping("/clear-cache")
    public ResponseEntity clearCache() {
        recommender.clearCache();
        return ResponseEntity.ok().body(null);
    }

}
