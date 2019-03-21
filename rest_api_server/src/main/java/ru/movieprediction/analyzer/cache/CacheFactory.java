package ru.movieprediction.analyzer.cache;

import org.ehcache.Cache;
import org.ehcache.CacheManager;
import org.ehcache.config.builders.CacheConfigurationBuilder;
import org.ehcache.config.builders.CacheManagerBuilder;
import org.ehcache.config.builders.ResourcePoolsBuilder;
import org.ehcache.expiry.Duration;
import org.ehcache.expiry.Expirations;
import org.springframework.stereotype.Component;
import ru.movieprediction.analyzer.model.RecommendationDTO;

import java.util.concurrent.TimeUnit;

@Component
public class CacheFactory {

    private static CacheManager cacheManager;

    private CacheFactory() {
    }

    static Cache<Integer, RecommendationDTO> getRecommendationCache(Integer heapSize, Integer duration) {
        if (cacheManager == null) {
            cacheManager = CacheManagerBuilder.newCacheManagerBuilder()
                    .withCache("recommendationCache",
                            CacheConfigurationBuilder.newCacheConfigurationBuilder(Integer.class, RecommendationDTO.class, ResourcePoolsBuilder.heap(1000))
                                    .withExpiry(Expirations.timeToLiveExpiration(Duration.of(60, TimeUnit.MINUTES))))
                    .build();
            cacheManager.init();
        }
        return cacheManager.getCache("recommendationCache", Integer.class, RecommendationDTO.class);
    }
}
