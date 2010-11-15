package com.scottbyrns.ml.neural.Learning;

import com.scottbyrns.ml.datasets.Pattern;
import com.scottbyrns.ml.neural.FeedForwardNeuralNetwork;
import com.scottbyrns.ml.neural.Neuron;
import com.scottbyrns.ml.neural.NeuronLayer;
import com.scottbyrns.ml.neural.Synapse;

import java.util.Iterator;
import java.util.Vector;

/**
 * @author Scott Byrns
 * Date: Nov 12, 2010
 * Time: 11:57:17 PM
 * @TODO finish javadocs
 *
 * @version 1.0
 */
public class BackPropagation extends AbstractFeedForwardNetworkLearningAlgorithm  {

	/**
	 * The algorithm's learning rate
	 */
	private static final double	DEFAULT_LEARNING_RATE	= 1.0;

	/**
	 * The algorithm's momentum rate
	 */
	private static final double	DEFAULT_MOMENTUM_RATE	= 0.1;

    /**
	 * The algorithm's learning rate determines the speed of training
	 */

	public double learningRate;

	/**
	 * The algorithm's learning rate determines the influence of the previous
	 * update on the current update
	 */
	public double momentumRate;

    /**
     * 
     * @param patterns The list of patterns the network is going to be trained with
     */
    @Override
    protected void trainEpoch(Vector<Pattern> patterns) {
        for (Pattern pattern : patterns) {
            getNetwork().feedForward(pattern.getInput());
            calculateDeltas(pattern.getOutput());
            adjustWeights();
        }
    }

    /**
     *
     * @param network
     * @param maximumEpochs
     * @param minimumError
     */
    public BackPropagation (FeedForwardNeuralNetwork network, int maximumEpochs, double minimumError) {
        super(network, maximumEpochs, minimumError);
        initilizeRates();
    }

    /**
     *
     * @param network
     */
    public BackPropagation (FeedForwardNeuralNetwork network) {
        super(network);
        initilizeRates();
    }

    /**
     *
     * @param network
     * @param maximumEpochs
     */
    public BackPropagation (FeedForwardNeuralNetwork network, int maximumEpochs) {
        super(network, maximumEpochs);
        initilizeRates();
    }

    /**
     * 
     * @param network
     * @param minimumError
     */
    public BackPropagation (FeedForwardNeuralNetwork network, double minimumError) {
        super(network, minimumError);
        initilizeRates();
    }

    /**
     * Set the learning and momentum rates to their default values.
     */
    private void initilizeRates () {
        setLearningRate(DEFAULT_LEARNING_RATE);
        setMomentumRate(DEFAULT_MOMENTUM_RATE);
    }

	/**
	 * Adjusts the network's weights
	 */
	private void adjustWeights()
	{
        Iterator<NeuronLayer> neuronLayersIteterator = getNetwork().getNeuronLayersIterator();
        while (neuronLayersIteterator.hasNext()) {
            /**
             * @TODO keep an eye on this, this used to exculde the last layer of the neuron vector
             * in the network. This may have been to avoid adjusting weights on the outputs.
             * If this has caused issues then get the next neuron layer and check if the iterator
             * hasNext after before adjusting the weights.
             */
            adjustWeights(neuronLayersIteterator.next());
        }
	}

	/**
	 * Adjusts the weights of the outgoing synapses this neuron feeds data into
     * 
     * @param neuron to adjust the outgoing synapse weight for.
	 */
	private void adjustWeights(Neuron neuron)
	{
        Iterator<Synapse> synapseIterator = neuron.getOutgoingSynapseIterator();
        Synapse synapse;
		while (synapseIterator.hasNext()) {
            synapse = synapseIterator.next();
			double weight = synapse.getWeight();
			double aWeightUpdate = neuron.getOutput() * getLearningRate() * synapse.getOutputNeuron().getDelta() + getMomentumRate() * getOldWeightUpdate(synapse);
			setWeightUpdate(synapse, aWeightUpdate);
			synapse.setWeight(weight + aWeightUpdate);
		}
	}

	/**
	 * Adjusts the weights of the outgoing synapses
     *
     * @param neuronLayer of neurons to adjust the outgoing synapse weights for.
	 */
	private void adjustWeights(NeuronLayer neuronLayer) {
        Iterator<Neuron> neuronIterator = neuronLayer.getNeuronsIterator();

        while (neuronIterator.hasNext()) {
            adjustWeights(neuronIterator.next());
        }
	}

    /**
     * Get the learningRate
     *
     * @return learningRate
     */
    private double getLearningRate() {
        return learningRate;
    }

    /**
     * Set the learningRate
     *
     * @param learningRate new learningRate
     */
    private void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Get the momentumRate
     *
     * @return momentumRate
     */
    private double getMomentumRate() {
        return momentumRate;
    }

    /**
     * Set the momentumRate.
     *
     * @param momentumRate new momentumRate
     */
    private void setMomentumRate(double momentumRate) {
        this.momentumRate = momentumRate;
    }
}
