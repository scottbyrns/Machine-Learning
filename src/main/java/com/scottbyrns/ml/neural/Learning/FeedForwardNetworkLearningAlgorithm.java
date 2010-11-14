package com.scottbyrns.ml.neural.Learning;

import com.scottbyrns.ml.datasets.PatternSet;
import com.scottbyrns.ml.neural.FeedForwardNeuralNetwork;

/**
 * Created by scott
 * Date: Nov 12, 2010
 * Time: 11:19:48 AM
 */
public interface FeedForwardNetworkLearningAlgorithm {
    /**
     * @TODO java doc this file.
     */

    public static final int DEFAULT_MAXIMUM_EPOCHS = 10000;
    public static final double DEFAULT_MINIMUM_ERROR = 0.001;

    /**
     * Get the current algorithm epoch.
     * @return current epoch
     */
    public int getCurrentEpoch();

    public void setPatternSet(PatternSet patternSet);

    public LearningStrategy getLearningStrategy();
    public void setLearningStrategy(LearningStrategy learningStrategy);

    public void start();
    public boolean isRunning();

    public FeedForwardNeuralNetwork getNetwork();

    public void setTargetError(double error);

}
