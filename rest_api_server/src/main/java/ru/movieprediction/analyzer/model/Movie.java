package ru.movieprediction.analyzer.model;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Movie {

    private Integer id;
    private Short year;
    private String name;
    private Float rating;
    private Float predictedRating;

}
