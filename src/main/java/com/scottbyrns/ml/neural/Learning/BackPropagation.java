package com.scottbyrns.ml.neural.Learning;

import com.scottbyrns.ml.datasets.Pattern;
import com.scottbyrns.ml.neural.FeedForwardNeuralNetwork;
import com.scottbyrns.ml.neural.Neuron;
import com.scottbyrns.ml.neural.NeuronLayer;
import com.scottbyrns.ml.neural.Synapse;

import java.util.Iterator;
import java.util.Vector;

/**
 * Created by scott
 * Date: Nov 12, 2010
 * Time: 11:57:17 PM
 */
public class BackPropagation extends AbstractFeedForwardNetworkLearningAlgorithm  {

    /**
     * TODO refactor to use getters / setters
     */
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
	public double				learning_rate			= DEFAULT_LEARNING_RATE;

	/**
	 * The algorithm's learning rate determines the influence of the previous
	 * update on the current update
	 */
	public double				momentum_rate			= DEFAULT_MOMENTUM_RATE;


    @Override
    protected void trainEpoch(Vector<Pattern> patterns) {
        for (Pattern pattern : patterns) {
            getNetwork().feedForward(pattern.getInput());
            calculateDeltas(pattern.getOutput());
            adjustWeights();
        }
    }

    public BackPropagation (FeedForwardNeuralNetwork network, int maximumEpochs, double minimumError) {
        super(network, maximumEpochs, minimumError);
    }

    public BackPropagation (FeedForwardNeuralNetwork network) {
        super(network);
    }

    public BackPropagation (FeedForwardNeuralNetwork network, int maximumEpochs) {
        super(network, maximumEpochs);
    }

    public BackPropagation (FeedForwardNeuralNetwork network, double minimumError) {
        super(network, minimumError);
    }

	/**
	 * Adjusts the network's weights
	 */
	private void adjustWeights()
	{
		for (int x = 0; x < (getNetwork().getNeuronLayers().size() - 1); x++)
		{
			adjustWeights(getNetwork().getNeuronLayers().get(x));
		}
	}

	/**
	 * Adjusts the weights of the outgoing synapses this neuron feeds data into
	 */
	private void adjustWeights(Neuron neuron)
	{
        Iterator<Synapse> synapseIterator = neuron.getOutgoingSynapseIterator();
        Synapse synapse;
		while (synapseIterator.hasNext()) {
            synapse = synapseIterator.next();
			double weight = synapse.getWeight();
			double a_weight_update = neuron.getOutput() * this.learning_rate * synapse.getOutputNeuron().getDelta() + this.momentum_rate * getOldWeightUpdate(synapse);
			setWeightUpdate(synapse, a_weight_update);
			synapse.setWeight(weight + a_weight_update);
		}
	}

	/**
	 * Adjusts the weights of the outgoing synapses
	 */
	private void adjustWeights(NeuronLayer layer)
	{
		for (int y = 0; y < layer.getNetworkSize(); y++)
		{
			adjustWeights(layer.getNeuron(y));
		}
	}
}
