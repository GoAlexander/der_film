package ru.movieprediction.analyzer.model;

import lombok.Getter;
import lombok.Setter;
import org.json.simple.JSONArray;

import java.time.LocalDateTime;

@Getter
@Setter
public class RecommendationDTO {

    private Integer userId;
    private Integer numOfMovies;
    private JSONArray movies;
    private LocalDateTime recommendTime;

}
