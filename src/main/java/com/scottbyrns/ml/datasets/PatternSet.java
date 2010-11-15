package com.scottbyrns.ml.datasets;

import com.scottbyrns.ml.helpers.IntervalScaler;

import java.util.List;

/**
 * @author Scott Byrns
 * Date: Nov 11, 2010
 * Time: 8:37:42 PM
 *
 * @version 1.0
 * @TODO finish java docs
 */
public interface PatternSet {

    /**
	 * The default percentage of patterns that belongs to the training set
	 */
    public static final double DEFAULT_PERCENTAGE_TRAINING_SET = 0.60;

    /**
	 * The default percentage of patterns that belongs to the validation set
	 */
    public static final double DEFAULT_PERCENTAGE_VALIDATION_SET = 0.30;

    /**
	 * The default percentage of patterns that belongs to the test set
	 */
    public static final double DEFAULT_PERCENTAGE_TEST_SET = 0.10;

    /**
     * Set custom percentages for the sets.
     * @param trainingSet
     * @param validationSet
     * @param testSet
     */
    public void setSetPercentages(double trainingSet, double validationSet, double testSet);

    /**
     * Set the training set percentage.
     * @param trainingSetPercentage
     */
    public void setTrainingSetPercentage (double trainingSetPercentage);

    /**
     * Set the validation set percentage.
     * @param validationSetPercentage
     */
    public void setValidationSetPercentage (double validationSetPercentage);

    /**
     * Set the test set percentage.
     * @param testSetPercentage
     */
    public void setTestSetPercentage (double testSetPercentage);

    /**
     * Add a pattern the to the pattern set.
     * @param pattern
     */
    public void addPattern (Pattern pattern);

    public List<Pattern> getTrainingSet();
    public List<Pattern> getValidationSet();
    public List<Pattern> getTestSet();

	/**
	 * Loads a pattern set from a file
	 *
	 * @param path Path to the pattern file
	 * @param input_size Number of inputs per line
     * @param splitToken the token used to delineate values in the pattern file.
	 * @return Number of parsed patterns, -1 in case of error
	 */
	public int loadPatterns(String path, int input_size, String splitToken);

    public List<Pattern> getShrunkPatterns (PatternType patternType);

    public IntervalScaler getInputInterval();
    public IntervalScaler getOutputInterval();
}
