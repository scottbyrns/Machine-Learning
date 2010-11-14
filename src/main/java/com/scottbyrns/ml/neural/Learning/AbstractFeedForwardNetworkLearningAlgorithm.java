package com.scottbyrns.ml.neural.Learning;

import com.scottbyrns.ml.datasets.Pattern;
import com.scottbyrns.ml.datasets.PatternSet;
import com.scottbyrns.ml.datasets.PatternType;
import com.scottbyrns.ml.neural.*;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

/**
 * @author Scott Byrns
 * Date: Nov 12, 2010
 * Time: 11:03:30 AM
 *
 * @version 1.0
 *
 * @TODO clean this shit up & javadoc
 */
public abstract class AbstractFeedForwardNetworkLearningAlgorithm extends Thread implements FeedForwardNetworkLearningAlgorithm {

    private boolean running = false;

    private FeedForwardNeuralNetwork network;

    private PatternSet patternSet;

    private int maximumEpochs;

    private int currentEpoch;

    private double minimumError, currentValidationError, currentTrainingError;

    private double minimumValidationError = Double.MAX_VALUE;
    private double minimumTrainingError = Double.MAX_VALUE - 1;

    private Vector<Double> minimumValidationErrorWeights;
    private double minimumValidationErrorEpoch;

    private Vector<Double> minimumTrainingErrorWeights;
    private double minimumTrainingErrorEpoch;


    private HashMap<Synapse, Double> deltaWeight, errorPartialDerivative, oldErrorPartialDerivative;
    /**
     * The previous weight update performed in each synapse
     */
    private HashMap<Synapse, Double> oldWeightUpdate;

    /**
     * The weight update to perform in each synapse
     */
    protected HashMap<Synapse, Double> weightUpdate;
    private LearningStrategy learningStrategy = LearningStrategy.Generalization;


    public AbstractFeedForwardNetworkLearningAlgorithm(FeedForwardNeuralNetwork network, int maximumEpochs, double minimumError) {
        setNetwork(network);
        setMaximumEpochs(maximumEpochs);
        setMinimumError(minimumError);

        reset();
    }

    public AbstractFeedForwardNetworkLearningAlgorithm(FeedForwardNeuralNetwork network) {
        this(network, FeedForwardNetworkLearningAlgorithm.DEFAULT_MAXIMUM_EPOCHS, FeedForwardNetworkLearningAlgorithm.DEFAULT_MINIMUM_ERROR);
    }

    public AbstractFeedForwardNetworkLearningAlgorithm(FeedForwardNeuralNetwork network, int maximumEpochs) {
        this(network, maximumEpochs, FeedForwardNetworkLearningAlgorithm.DEFAULT_MINIMUM_ERROR);
    }

    public AbstractFeedForwardNetworkLearningAlgorithm(FeedForwardNeuralNetwork network, double minimumError) {
        this(network, FeedForwardNetworkLearningAlgorithm.DEFAULT_MAXIMUM_EPOCHS, minimumError);
    }

    @Override
    public void run() {
        setRunning(true);

        if (getLearningStrategy() == LearningStrategy.Memorize) {
            memorize(getPatternSet().getShrunkPatterns(PatternType.All));
        } else if (getLearningStrategy() == LearningStrategy.Generalization) {
            generalize(getPatternSet().getShrunkPatterns(PatternType.Training), getPatternSet().getShrunkPatterns(PatternType.Validation));
        }

        setRunning(false);
    }

    /**
     * Get the current algorithm epoch.
     *
     * @return current epoch
     */
    public int getCurrentEpoch() {
        return currentEpoch;
    }

    /**
     * The set of patterns to train the network with.
     *
     * @param patternSet to train the network.
     */
    public void setPatternSet(PatternSet patternSet) {
        this.patternSet = patternSet;
    }

    /**
     * Get the LearningStrategy of the learning algorithm.
     *
     * @return LearningStrategy of the learning algorithm.
     */
    public LearningStrategy getLearningStrategy() {
        return learningStrategy;
    }

    /**
     * Set the learning strategy of the learning algorithm to the specified
     * LearningStrategy
     *
     * @param learningStrategy to use in the learning algorithm
     */
    public void setLearningStrategy(LearningStrategy learningStrategy) {
        this.learningStrategy = learningStrategy;
    }

    /**
     * Start training the neural network.
     */
    public void startTraining () {
        start();
    }

    /**
     * Is the network currently training?
     *
     * @return boolean representation of execution state.
     */
    public boolean isRunning() {
        return running;
    }

    /**
     * Get the FeedForwardNeuralNetwork being trained.
     *
     * @return FeedForwardNeuralNetwork being trained
     */
    public FeedForwardNeuralNetwork getNetwork() {
        return network;
    }

    /**
     * Set the target error for training.
     *
     * @param error target
     */
    public void setTargetError(double error) {
        setMinimumError(error);
    }














    /**
     * Resets the partial derivative field of all synapses.
     *
     * @return Boolean indicating if the operation was successful
     */
    protected boolean resetPartialDerivatives() {
        try {
            Iterator<SynapseLayer> synapseLayerIterator = getNetwork().getSynapseLayersIterator();
            Iterator<Synapse> synapseIterator;
            Synapse synapse;

            while (synapseLayerIterator.hasNext()) {
                synapseIterator = synapseLayerIterator.next().getSynapsesIterator();
                while (synapseIterator.hasNext()) {
                    synapse = synapseIterator.next();
                    getErrorPartialDerivative().put(synapse, 0.0);
                    getOldErrorPartialDerivative().put(synapse, 0.0);
                    getDeltaWeight().put(synapse, 0.1);
                }
            }

            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Trains the network to generalize by the "Early Stopping Method of
     * Training
     *
     * @param trainingPatterns   Training set
     * @param validationPatterns Validation set
     * @return Boolean indicating if the operation was successful
     */
    private boolean generalize(Vector<Pattern> trainingPatterns, Vector<Pattern> validationPatterns) {
        try {

            reset();

            while (getCurrentEpoch() < getMaximumEpochs() && getCurrentValidationError() >= getMinimumError()) {

                trainNextEpoch(trainingPatterns);

                setCurrentTrainingError(getNetwork().getPredictionError(trainingPatterns));
                setCurrentValidationError(getNetwork().getPredictionError(validationPatterns));

                if (getCurrentValidationError() < getMinimumValidationError()) {
                    setMinimumValidationErrorWeights(getNetwork().getWeightVector());
                    setMinimumValidationError(getCurrentValidationError());
                    setMinimumValidationErrorEpoch(getCurrentEpoch());
                }

                if (getCurrentTrainingError() < getMinimumTrainingError()) {
                    setMinimumTrainingErrorWeights(getNetwork().getWeightVector());
                    setMinimumTrainingError(getCurrentTrainingError());
                    setMinimumTrainingErrorEpoch(getCurrentEpoch());
                }

            }

            getNetwork().setWeightVector(getMinimumValidationErrorWeights());

            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Teaches the training set to the network by memorization
     *
     * @param patterns Training set
     * @return Boolean indicating if the operation was successful
     */
    protected boolean memorize(Vector<Pattern> patterns) {
        try {

            reset();

            do {

                trainNextEpoch(patterns);
                setCurrentTrainingError(getNetwork().getPredictionError(patterns));

                if (getCurrentTrainingError() < getMinimumTrainingError()) {
                    setMinimumTrainingErrorWeights(getNetwork().getWeightVector());
                    setMinimumTrainingError(getCurrentTrainingError());
                    setMinimumTrainingErrorEpoch(getCurrentEpoch());
                }

            }
            while (getCurrentTrainingError() > getMinimumError() && getCurrentEpoch() < getMaximumEpochs());

            getNetwork().setWeightVector(getMinimumTrainingErrorWeights());

            return true;
        }
        catch (RuntimeException e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Calculates the deltas for all the neurons
     *
     * @param output The desired network output
     * @return Boolean indicating if the operation was successful
     */
    protected boolean calculateDeltas(Vector<Double> output) {
        try {
            calculateOutputLayerDeltas(output);

            Iterator<NeuronLayer> hiddenLayersIterator = getNetwork().getHiddenNeuronLayersIterator();

            while (hiddenLayersIterator.hasNext()) {
                calculateHiddenLayerDeltas(hiddenLayersIterator.next());
            }

            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * @param synapse
     * @return the new derivative value. This function assumes that the neruon
     *         delta values (the neuron sensitivities) have already been
     *         updated; that is, that calculateDeltas(output) has already been
     *         called this epoch.
     */
    protected static double calculateErrorPartialDerivative(Synapse synapse) {
        double aDelta = synapse.getOutputNeuron().getDelta();
        double synapseValue = synapse.getInputNeuron().getOutput();
        return (-aDelta) * synapseValue;
    }

    /**
     * Returns a HashMap of old weight updates.
     *
     * @return HashMap of old weight updates.
     */
    protected HashMap<Synapse, Double> getOldWeightUpdate() {
        return oldWeightUpdate;
    }

    /**
     * Returns the old weight update for the given synapse
     *
     * @param synapse The synapse associated with the old weight update you want to retrieve
     * @return Returns the value of the previous update, returns 0 if it doesn't exist
     */
    protected double getOldWeightUpdate(Synapse synapse) {
        if (getOldWeightUpdate().get(synapse) == null) {
            return 0.0;
        }
        return getOldWeightUpdate().get(synapse);
    }

    protected HashMap<Synapse, Double> getWeightUpdate() {
        return weightUpdate;
    }

    protected HashMap<Synapse, Double> getDeltaWeight() {
        return deltaWeight;
    }

    protected void setDeltaWeight(HashMap<Synapse, Double> deltaWeight) {
        this.deltaWeight = deltaWeight;
    }

    protected HashMap<Synapse, Double> getErrorPartialDerivative() {
        return errorPartialDerivative;
    }

    protected void setErrorPartialDerivative(HashMap<Synapse, Double> errorPartialDerivative) {
        this.errorPartialDerivative = errorPartialDerivative;
    }

    protected HashMap<Synapse, Double> getOldErrorPartialDerivative() {
        return oldErrorPartialDerivative;
    }

    protected void setOldErrorPartialDerivative(HashMap<Synapse, Double> oldErrorPartialDerivative) {
        this.oldErrorPartialDerivative = oldErrorPartialDerivative;
    }

    protected PatternSet getPatternSet() {
        return patternSet;
    }

    /**
     * Updates the weight update for a given synapse, if one was already stored
     * then it is moved to the old weight update, and the new one takes it's
     * place
     *
     * @param synapse The synapse associated with the given weight update
     * @param update  The value of the update
     */
    protected void setWeightUpdate(Synapse synapse, double update) {
        Double stored_update = getWeightUpdate().get(synapse);
        if (stored_update != null) {
            getOldWeightUpdate().put(synapse, stored_update);
        }
        getWeightUpdate().put(synapse, update);
    }

    /**
     * Calculates the layer's derived error via the next layer's neuron's error *
     *
     * @param layer
     * @return Boolean indicating if the operation was successful
     */
    private boolean calculateHiddenLayerDeltas(NeuronLayer layer) {
        try {
            Iterator<Neuron> neuronIterator = layer.getNeuronsIterator();
            Neuron neuron;

            while (neuronIterator.hasNext()) {
                neuron = neuronIterator.next();
                neuron.setDelta(calculateHiddenNeuronDelta(neuron));
            }

            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }


    /**
     * Calculates the neuron's error via the error of the next layer's neurons
     *
     * @param neuron
     * @return double
     */
    private double calculateHiddenNeuronDelta(Neuron neuron) {
        double sum = 0;
        Iterator<Synapse> neuronIterator = neuron.getOutgoingSynapseIterator();

        while (neuronIterator.hasNext()) {
            Synapse synapse = neuronIterator.next();
            double downstreamDelta = synapse.getOutputNeuron().getDelta();
            sum += synapse.getWeight() * downstreamDelta;
        }

        double derivative = neuron.calculateDerivative(neuron.getInput());
        return derivative * sum;
    }

    /**
     * Calculate an output layer's error
     *
     * @param output Desired output
     * @return Boolean indicating if the operation was successful
     */
    private boolean calculateOutputLayerDeltas(Vector<Double> output) {
        try {
            Iterator<Neuron> outputLayerIterator = getNetwork().getOutputNeuronLayerIterator();
            Iterator<Double> outputDoubleIterator = output.iterator();
            Neuron neuron;

            while (outputLayerIterator.hasNext() && outputDoubleIterator.hasNext()) {
                neuron = outputLayerIterator.next();
                neuron.setDelta(calculateOutputNeuronDelta(outputDoubleIterator.next(), neuron));
            }

            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }


    /**
     * Calculates the neuron's error and stores it in the delta variable
     *
     * @param target Desired output
     * @param neuron
     * @return double
     */
    private double calculateOutputNeuronDelta(double target, Neuron neuron) {
        double derivative = neuron.calculateDerivative(neuron.getOutput());
        double error = target - neuron.getOutput();
        return derivative * error;
    }






    /**
     * Resets all the data used by the learning algorithm
     */
    private void reset() {

        setDeltaWeight(new HashMap<Synapse, Double>());
        setErrorPartialDerivative(new HashMap<Synapse, Double>());
        setOldErrorPartialDerivative(new HashMap<Synapse, Double>());
        setOldWeightUpdate(new HashMap<Synapse, Double>());
        setWeightUpdate(new HashMap<Synapse, Double>());

        setCurrentEpoch(0);

        setCurrentValidationError(Double.MAX_VALUE);
        setCurrentTrainingError(Double.MAX_VALUE);
        setMinimumTrainingErrorEpoch(0);
        setMinimumTrainingErrorWeights(null);
        setMinimumValidationErrorEpoch(0);
        setMinimumValidationError(Double.MAX_VALUE);
        setMinimumValidationErrorWeights(null);

        Iterator<NeuronLayer> neuronLayerIterator = getNetwork().getNeuronLayersIterator();
        NeuronLayer neuronLayer;
        Neuron neuron;
        Synapse synapse;
        Iterator<Neuron> neuronIterator;
        Iterator<Synapse> synapseIterator;

        while (neuronLayerIterator.hasNext()) {
            neuronLayer = neuronLayerIterator.next();
            neuronIterator = neuronLayer.getNeuronsIterator();
            while (neuronIterator.hasNext()) {
                neuron = neuronIterator.next();
                synapseIterator = neuron.getOutgoingSynapseIterator();
                while (synapseIterator.hasNext()) {
                    synapse = synapseIterator.next();
                    getWeightUpdate().put(synapse, 0D);
                    getOldErrorPartialDerivative().put(synapse, 0D);
                    getErrorPartialDerivative().put(synapse, 0D);
                    getDeltaWeight().put(synapse, 0D);
                }
            }
        }

        resetPartialDerivatives();
    }




    /*
     * Getters / Setters
     */

    /**
     * Set the running state of the network training.
     *
     * @param running boolean state
     */
    private void setRunning(boolean running) {
        this.running = running;
    }

    /**
     * Set the FeedForwardNeuralNetwork to be trained.
     *
     * @param network FeedForwardNeuralNetwork to be trained.
     */
    private void setNetwork(FeedForwardNeuralNetwork network) {
        this.network = network;
    }

    /**
     * Get the maximum number of epochs the training process may run.
     *
     * @return maximum number of epochs the training process may run.
     */
    private int getMaximumEpochs() {
        return maximumEpochs;
    }

    /**
     * Set the maximum number of epochs the training process may run.
     *
     * @param maximumEpochs the training process may run.
     */
    private void setMaximumEpochs(int maximumEpochs) {
        this.maximumEpochs = maximumEpochs;
    }

    /**
     * Get the minimum training error
     *
     * @return minimum training error
     */
    private double getMinimumError() {
        return minimumError;
    }

    /**
     * Set the minimum training error.
     *
     * @param minimumError minimum training error
     */
    private void setMinimumError(double minimumError) {
        this.minimumError = minimumError;
    }

    private void setCurrentEpoch(int currentEpoch) {
        this.currentEpoch = currentEpoch;
    }

    private void incrementEpoch() {
        setCurrentEpoch(getCurrentEpoch() + 1);
    }

    private double getCurrentValidationError() {
        return currentValidationError;
    }

    private void setCurrentValidationError(double currentValidationError) {
        this.currentValidationError = currentValidationError;
    }

    private double getCurrentTrainingError() {
        return currentTrainingError;
    }

    private void setCurrentTrainingError(double currentTrainingError) {
        this.currentTrainingError = currentTrainingError;
    }

    private double getMinimumValidationError() {
        return minimumValidationError;
    }

    private void setMinimumValidationError(double minimumValidationError) {
        this.minimumValidationError = minimumValidationError;
    }

    private double getMinimumTrainingError() {
        return minimumTrainingError;
    }

    private void setMinimumTrainingError(double minimumTrainingError) {
        this.minimumTrainingError = minimumTrainingError;
    }

    private Vector<Double> getMinimumValidationErrorWeights() {
        return minimumValidationErrorWeights;
    }

    private void setMinimumValidationErrorWeights(Vector<Double> minimumValidationErrorWeights) {
        this.minimumValidationErrorWeights = minimumValidationErrorWeights;
    }

    private double getMinimumValidationErrorEpoch() {
        return minimumValidationErrorEpoch;
    }

    private void setMinimumValidationErrorEpoch(double minimumValidationErrorEpoch) {
        this.minimumValidationErrorEpoch = minimumValidationErrorEpoch;
    }

    private Vector<Double> getMinimumTrainingErrorWeights() {
        return minimumTrainingErrorWeights;
    }

    private void setMinimumTrainingErrorWeights(Vector<Double> minimumTrainingErrorWeights) {
        this.minimumTrainingErrorWeights = minimumTrainingErrorWeights;
    }

    private double getMinimumTrainingErrorEpoch() {
        return minimumTrainingErrorEpoch;
    }

    private void setMinimumTrainingErrorEpoch(double minimumTrainingErrorEpoch) {
        this.minimumTrainingErrorEpoch = minimumTrainingErrorEpoch;
    }

    private void setOldWeightUpdate(HashMap<Synapse, Double> oldWeightUpdate) {
        this.oldWeightUpdate = oldWeightUpdate;
    }

    private void setWeightUpdate(HashMap<Synapse, Double> weightUpdate) {
        this.weightUpdate = weightUpdate;
    }

    
    /**
     * Wrap train epoch to take care of incrementing the epoch so the implementer doesnt have to.
     *
     * @param patterns
     */
    private void trainNextEpoch(Vector<Pattern> patterns) {
        try {
            trainEpoch(patterns);
        }
        catch (RuntimeException e) {
            e.printStackTrace();
        }
        incrementEpoch();
    }

    /**
     * Trains the neural network with a pattern for one epoch
     *
     * @param patterns The list of patterns the network is going to be trained with
     *                 for one epoch
     */
    protected abstract void trainEpoch(Vector<Pattern> patterns);

}
