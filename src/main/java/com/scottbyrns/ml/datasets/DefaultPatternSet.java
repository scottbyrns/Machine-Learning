package com.scottbyrns.ml.datasets;

import com.scottbyrns.ml.helpers.IntervalScaler;

import java.io.*;
import java.util.ArrayList;
import java.util.List;


/**
 * @author Scott Byrns
 * Date: Nov 11, 2010
 * Time: 8:38:20 PM
 *
 * @version 1.0
 * @TODO java doc and clean up
 */
public class DefaultPatternSet implements PatternSet {

    private boolean setsAreGenerated = false;

    private IntervalScaler outputInterval;

    private IntervalScaler inputInterval;

    private double trainingSetPercentage, validationSetPercentage, testSetPercentage;

    private List<Pattern> patterns, trainingSet, validationSet, testSet;

    public DefaultPatternSet() {

        setPatterns(new ArrayList<Pattern>());
        setTrainingSet(new ArrayList<Pattern>());
        setValidationSet(new ArrayList<Pattern>());
        setTestSet(new ArrayList<Pattern>());

        setInputInterval(new IntervalScaler(0, 1));
        setOutputInterval(new IntervalScaler(0, 1));

        setSetPercentages(
                PatternSet.DEFAULT_PERCENTAGE_TRAINING_SET,
                PatternSet.DEFAULT_PERCENTAGE_VALIDATION_SET,
                PatternSet.DEFAULT_PERCENTAGE_TEST_SET
        );
    }

    /**
     * Add a pattern the to the pattern set.
     * @param pattern
     */
    public void addPattern (Pattern pattern) {
        patterns.add(pattern);
    }

    /**
     * Set custom percentages for the sets.
     * @param trainingSetPercentage
     * @param validationSetPercentage
     * @param testSetPercentage
     */
    public void setSetPercentages(double trainingSetPercentage, double validationSetPercentage, double testSetPercentage) {
        setTrainingSetPercentage(trainingSetPercentage);
        setValidationSetPercentage(validationSetPercentage);
        setTestSetPercentage(testSetPercentage);
    }

    /**
     * Set the training set percentage.
     * @param trainingSetPercentage
     */
    public void setTrainingSetPercentage (double trainingSetPercentage) {
        this.trainingSetPercentage = trainingSetPercentage;
    }

    /**
     * Set the validation set percentage.
     * @param validationSetPercentage
     */
    public void setValidationSetPercentage (double validationSetPercentage) {
        this.validationSetPercentage = validationSetPercentage;
    }

    /**
     * Set the test set percentage.
     * @param testSetPercentage
     */
    public void setTestSetPercentage (double testSetPercentage) {
        this.testSetPercentage = testSetPercentage;
    }

	/**
	 * Generates the training, validation and test sets from the list of
	 * patterns
	 *
	 * @return Boolean indicating if the operation was successful
	 */
    private boolean generateSets () {
        if (isSetsAreGenerated()) {
            return true;
        }
        setSetsAreGenerated(true);
        try {

            int numberInTrainingSet, numberInValidationSet, numberInTestSet, position;

            numberInTrainingSet = patternSetSizeAsPercentageOfPatterns(getTrainingSetPercentage());
            numberInValidationSet = patternSetSizeAsPercentageOfPatterns(getValidationSetPercentage());
            numberInTestSet = patternSetSizeAsPercentageOfPatterns(getTestSetPercentage());

            position = 0;

            while (numberInTrainingSet-- > 0) {
                getTrainingSet().add(getPatterns().get(position));
                position += 1;
            }

            while (numberInValidationSet-- > 0) {
                getValidationSet().add(getPatterns().get(position));
                position += 1;
            }

            while (numberInTestSet-- > 0) {
                getTestSet().add(getPatterns().get(position));
                position += 1;
            }

            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Calculate the number of sets represent the specified percentage
     * of the total number of sets in the patterns.
     * @param percentage
     * @return
     */
    private int patternSetSizeAsPercentageOfPatterns (double percentage) {
        return (int) (percentage * getPatterns().size());
    }

    private List<Pattern> getPatterns() {
        return patterns;
    }

    private void setPatterns(List<Pattern> patterns) {
        this.patterns = patterns;
    }

    public List<Pattern> getTrainingSet() {
        generateSets();
        return trainingSet;
    }

    private void setTrainingSet(List<Pattern> trainingSet) {
        this.trainingSet = trainingSet;
    }

    public List<Pattern> getValidationSet() {
        generateSets();
        return validationSet;
    }

    private void setValidationSet(List<Pattern> validationSet) {
        this.validationSet = validationSet;
    }

    public List<Pattern> getTestSet() {
        generateSets();
        return testSet;
    }

    private void setTestSet(List<Pattern> testSet) {
        this.testSet = testSet;
    }

    private double getTrainingSetPercentage() {
        return trainingSetPercentage;
    }

    private double getValidationSetPercentage() {
        return validationSetPercentage;
    }

    private double getTestSetPercentage() {
        return testSetPercentage;
    }

    public boolean isSetsAreGenerated() {
        return setsAreGenerated;
    }

    public void setSetsAreGenerated(boolean setsAreGenerated) {
        this.setsAreGenerated = setsAreGenerated;
    }

	/**
	 * Loads a pattern set from a file
	 *
	 * @param path Path to the pattern file
	 * @param input_size Number of inputs per line
     * @param splitToken the token used to delineate values in the pattern file.
	 * @return Number of parsed patterns, -1 in case of error
	 */
	public int loadPatterns(String path, int input_size, String splitToken) {
        try
        {
            /**
             * @TODO refactor
             */
            BufferedReader leitor = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = leitor.readLine()) != null)
            {
                int input = 0;
                int x = 0;
                for (x = 0; x < line.length() && input < input_size; x++)
                    if (new String("" + line.charAt(x)).equals(splitToken))
                        input++;
                if (input == input_size)
                {
                    String input_string = line.substring(0, x - 1);
                    String output_string = line.substring(x);
                    Pattern pattern = new DefaultPattern(input_string, output_string, splitToken);
                    getPatterns().add(pattern);
                }
            }
//            setPatternSetRanges();
            return getPatterns().size();
        }
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
            System.err.println("ERROR: File not found in " + path);
            return -1;
        }
        catch (IOException e)
        {
            e.printStackTrace();
            System.err.println("ERROR: Error reading file in " + path);
            return -1;
        }
	}

    public List<Pattern> getShrunkPatterns (PatternType patternType) {
        if (patternType == PatternType.All) {
            return getPatterns();
        }
        else if (patternType == PatternType.Test) {
            return getTestSet();
        }
        else if (patternType == PatternType.Training) {
            return getTrainingSet();
        }
        else if (patternType == PatternType.Validation) {
            return getValidationSet();
        }
        else {
            return null;
        }
    }

    public IntervalScaler getOutputInterval() {
        return outputInterval;
    }

    public void setOutputInterval(IntervalScaler outputInterval) {
        this.outputInterval = outputInterval;
    }

    public IntervalScaler getInputInterval() {
        return inputInterval;
    }

    public void setInputInterval(IntervalScaler inputInterval) {
        this.inputInterval = inputInterval;
    }
    
}
