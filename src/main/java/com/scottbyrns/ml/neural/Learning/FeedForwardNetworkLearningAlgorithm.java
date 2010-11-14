package com.scottbyrns.ml.neural.Learning;

import com.scottbyrns.ml.datasets.PatternSet;
import com.scottbyrns.ml.neural.FeedForwardNeuralNetwork;

/**
 * @author Scott Byrns
 * Date: Nov 12, 2010
 * Time: 11:19:48 AM
 *
 * @version 1.0
 */
public interface FeedForwardNetworkLearningAlgorithm {

    /**
     * Default maximum training epochs.
     */
    public static final int DEFAULT_MAXIMUM_EPOCHS = 10000;

    /**
     * Default minimum training error.
     */
    public static final double DEFAULT_MINIMUM_ERROR = 0.001;

    /**
     * Get the current algorithm epoch.
     * 
     * @return current epoch
     */
    public int getCurrentEpoch();

    /**
     * The set of patterns to train the network with.
     *
     * @param patternSet to train the network.
     */
    public void setPatternSet(PatternSet patternSet);

    /**
     * Get the LearningStrategy of the learning algorithm.
     *
     * @return LearningStrategy of the learning algorithm.
     */
    public LearningStrategy getLearningStrategy();

    /**
     * Set the learning strategy of the learning algorithm to the specified
     * LearningStrategy
     *
     * @param learningStrategy to use in the learning algorithm
     */
    public void setLearningStrategy(LearningStrategy learningStrategy);

    /**
     * Start training the neural network.
     */
    public void startTraining();

    /**
     * Is the network currently training?
     * 
     * @return boolean representation of execution state.
     */
    public boolean isRunning();

    /**
     * Get the FeedForwardNeuralNetwork being trained.
     *
     * @return FeedForwardNeuralNetwork being trained 
     */
    public FeedForwardNeuralNetwork getNetwork();

    /**
     * Set the target error for training.
     *
     * @param error target
     */
    public void setTargetError(double error);

}
