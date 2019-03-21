package ru.movieprediction.analyzer.cache;

import org.ehcache.Cache;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import ru.movieprediction.analyzer.model.RecommendationDTO;

@Configuration
public class CacheConfiguration {

    @Bean
    public Cache<Integer, RecommendationDTO> getCache(@Value("${predictions.caching.expire.minutes}") Integer duration,  @Value("${predictions.heap.size}") Integer heapSize) {
        return CacheFactory.getRecommendationCache(heapSize, duration);
    }

}
