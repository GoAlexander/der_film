package ru.movieprediction.analyzer.engine;

public enum PredictionScript {
    FAVORITES("dl_film_prediction.py"),
    //TODO Rename according to LupusSanctus suggestions
    CLUSTERISATION("forecasting_words_line.py");

    private final String scriptName;

    PredictionScript(String scriptName) {
        this.scriptName = scriptName;
    }

    public String getScriptName() {
        return scriptName;
    }
}
