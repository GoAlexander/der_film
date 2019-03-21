package ru.movieprediction.analyzer.engine;

import org.apache.log4j.Logger;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

@Component
class Executor {

    private static final Logger LOGGER = Logger.getLogger(Executor.class);

    private static String pathToPython;

    @Value("${path.to.python}")
    public void setPathToPython(String pathToPython) {
        Executor.pathToPython = pathToPython;
    }

    static String executePredictor(PredictionScript predictionScript, String userId, String amount) {
        String s;
        StringBuilder stdout = new StringBuilder();
        StringBuilder stderr = new StringBuilder();

        try{
            Process p = Runtime.getRuntime().exec(pathToPython+ " deep_learning/" + predictionScript.getScriptName() + " --data-path=data --user-id=" + userId + " --recommend=" + amount);

            BufferedReader stdInput = new BufferedReader(new
                    InputStreamReader(p.getInputStream()));

            BufferedReader stdError = new BufferedReader(new
                    InputStreamReader(p.getErrorStream()));

            // read the output from the command
            //System.out.println("Executions results:\n");
            while ((s = stdInput.readLine()) != null) {
                stdout.append(s).append(System.getProperty("line.separator"));
                //System.out.println(s);
            }

            while ((s = stdError.readLine()) != null) {
                stderr.append(s).append(System.getProperty("line.separator"));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        //TODO Check issue with logs in terminal window
        LOGGER.debug("Output of script:\n" + stdout.toString());
        LOGGER.debug("Errors & warnings of execution (if any):\n" + stderr.toString());
        return stdout.toString();
    }
}
