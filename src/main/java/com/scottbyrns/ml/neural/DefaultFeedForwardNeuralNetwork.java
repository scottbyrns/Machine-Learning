package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.Mathematics;
import com.scottbyrns.ml.datasets.Pattern;
import com.scottbyrns.ml.neural.Activation.ActivationFunction;
import com.scottbyrns.ml.neural.Activation.ActivationFunctionLinear;
import com.scottbyrns.ml.neural.Activation.ActivationFunctionSigmoid;

import java.util.Iterator;
import java.util.Vector;

/**
 * Default implementation of the FeedForwardNeuralNetwork interface.
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 2:10:48 PM
 */
public class DefaultFeedForwardNeuralNetwork implements FeedForwardNeuralNetwork {

    public static final int DEFAULT_NUMBER_BIAS_NEURONS = 1;

    private Vector<NeuronLayer> neuronLayers;
    private Vector<SynapseLayer> synapseLayers;

    /**
     * Create an instance of a DefaultFeedForwardNeuralNetwork from an existing FeedForwardNeuralNetwork
     * @param feedForwardNeuralNetwork
     */
    public DefaultFeedForwardNeuralNetwork(FeedForwardNeuralNetwork feedForwardNeuralNetwork) {
        /* Sorry for the formatting, this method call was way too long visually otherwise */
        this(
                feedForwardNeuralNetwork.getNumberNeuronsInput(NeuronType.Normal),
                feedForwardNeuralNetwork.getNumberNeuronsHidden(NeuronType.Normal),
                feedForwardNeuralNetwork.getNumberNeuronsOutput(NeuronType.Normal)
        );
    }

    /**
     * Create a new DefaultFeedForwardNeuralNetwork
     * @param inputSize number of inputs
     * @param hiddenSizes array of hidden layer neuron counts
     * @param outputSize number of outputs
     */
    public DefaultFeedForwardNeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize) {
        this(inputSize, hiddenSizes, outputSize, new ActivationFunctionSigmoid());
    }

    /**
     * Create a new DefaultFeedForwardNeuralNetwork
     * @param inputSize number of inputs
     * @param hiddenSizes array of hidden layer neuron counts
     * @param outputSize number of outputs
     * @param activationFunction neuron activation function.
     */
    public DefaultFeedForwardNeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, ActivationFunction activationFunction) {
        try {
			// Create the layers
			int bias_nodes_per_layer = DefaultFeedForwardNeuralNetwork.DEFAULT_NUMBER_BIAS_NEURONS;
			setNeuronLayers(new Vector<NeuronLayer>());
			getNeuronLayers().add(new DefaultNeuronLayer(inputSize, new ActivationFunctionLinear(), bias_nodes_per_layer));
			for (int x = 0; x < hiddenSizes.length; x++) {
                getNeuronLayers().add(new DefaultNeuronLayer(hiddenSizes[x], activationFunction, bias_nodes_per_layer));
            }

			getNeuronLayers().add(new DefaultNeuronLayer(outputSize, activationFunction));
			// Connect the layers
			setSynapseLayers(new Vector<SynapseLayer>());
			for (int x = 0; x < getNeuronLayers().size() - 1; x++) {
                connectLayers(x, x + 1);
            }

        }
        catch (RuntimeException e) {

        }
    }

    /**
	 * Returns the predicted pattern for an input
	 *
	 * @param input The input
	 * @return The predicted pattern for the given input, or null in case of
	 *         error
	 */
	public Vector<Double> getPrediction(Vector<Double> input) {
        feedForward(input);
        return getOutput();
    }

	/**
	 * Returns the network's output
	 *
	 * @return Returns an array of doubles with the network's output
	 */
	private Vector<Double> getOutput() {
		Vector<Double> output = new Vector<Double>();
		for (int x = 0; x < getNeuronLayers().get(getNeuronLayers().size() - 1).getNetworkSize(); x++)
			output.add(new Double(getNeuronLayers().get(getNeuronLayers().size() - 1).getNeuron(x).getOutput()));
		return output;
	}

    /**
     * Connect two neurons together and return the connecting synapse.
     * @param source
     * @param destination
     * @return The connecting neuron or null if something went wrong.
     */
    public Synapse connectNeurons(Neuron source, Neuron destination) {

        try {
            double weight = Mathematics.rand();
            Synapse synapse = new DefaultSynapse(source, destination, weight);

            source.addOutgoingSynapse(synapse);
            destination.addIncomingSynapse(synapse);

            return synapse;
        }
        catch (RuntimeException e) {
            return null;
        }

    }

	/**
	 * Gets a vector of weights contained in the neural network
	 *
	 * @return Vector of weight values
	 */
	public Vector<Double> getWeightVector() {
        Vector<Double> list = new Vector<Double>();
        for (SynapseLayer synapseLayer: getSynapseLayers()) {
            for (Double value : synapseLayer.getWeightVector()) {
                list.add(value);
            }
        }
        return list;   
    }

    /**
     * Set the weights in the network to the values of the provided vector.
     *
     * @param weightVector Vector of doubles to populate the synapses.
     */
    public void setWeightVector (Vector<Double> weightVector) {
        
        Iterator<SynapseLayer> synapseLayerIterator = getSynapseLayers().iterator();
        Iterator<Double> weightVectorIterator = weightVector.iterator();
        
        while (synapseLayerIterator.hasNext()) {
            synapseLayerIterator.next().setWeightVector(weightVectorIterator);
        }
    }

	/**
	 * Returns the mean squared error for a input-output pair
	 *
	 * @param input The input
	 * @param output The desired output
	 * @return The mean squared error for the input-output pair, or -1.0 in case
	 *         of error
	 */
	public double getPredictionError(Vector<Double> input, Vector<Double> output) {
        try {
            feedForward(input);
            return meanSquaredError(output);
        }
        catch (RuntimeException e) {
            return -1.0;
        }
    }

	/**
	 * Returns the mean squared error for a set of input-output pairs
	 *
	 * @param patterns A vector of input-output pairs
	 * @return The mean squared error for the set of input-output pairs, or -1.0
	 *         in case of error
	 */
	public double getPredictionError(Vector<Pattern> patterns) {
        try {

            double error = 0;
            for (Pattern pattern : patterns) {
                error += getPredictionError(pattern.getInput(), pattern.getOutput());
            }
            return error / patterns.size();

        }
        catch (RuntimeException e) {
            return -1.0;
        }
    }

    /**
     * Returns the current error for the a vector of patterns
     *
     * @param patterns The list of patterns that will be tested
     * @return The mean squared error for the list of patterns, or -1.0 in case of error
     */
    public double measurePatternListError(Vector<Pattern> patterns) {
        try {
            double error = 0;
			int count = 0;
			for (Pattern pattern : patterns) {
				feedForward(pattern.getInput());
				error += meanSquaredError(pattern.getOutput());
				count++;
			}
			error /= count;
			return error;
        }
        catch (RuntimeException e) {
            return -1.0;
        }
    }

	/**
	 * Returns the mean squared error between the network's output and the
	 * desired one
	 *
	 * @param output Desired output
	 * @return Returns the mean squared error, or -1.0 in case of error
	 */
    public double meanSquaredError(Vector<Double> output) {
        try {
			double sum = 0;
			int lastLayerIndex = getNeuronLayers().size() - 1;
			NeuronLayer lastLayer = getNeuronLayers().get(lastLayerIndex);
			for (int x = 0; x < output.size(); x++)
			{
				double network_output = lastLayer.getNeuron(x).getOutput();
				double desired_output = output.get(x).doubleValue();
				sum += Math.pow(desired_output - network_output, 2);
			}
			return sum / 2;
        }
        catch (RuntimeException e) {
            return -1.0;
        }
    }

	/**
	 * Calculates the network's output by feeding the input all the way to the
	 * output layer
	 *
	 * @param input The network's input
	 * @return Returns the network's output, or null in case of error
	 */
    public Vector<Double> feedForward(Vector<Double> input) {
        try {
			resetValues();
			feedForwardInputLayer(input);
			getNeuronLayers().get(0).feedForward();
			for (int x = 1; x < (getNeuronLayers().size() - 1); x++)
			{
				getNeuronLayers().get(x).calculateOutput();
				getNeuronLayers().get(x).feedForward();
			}
			int output_layer_index = getNeuronLayers().size() - 1;
			getNeuronLayers().get(output_layer_index).calculateOutput();
			return getNeuronLayers().get(output_layer_index).getOutput();
        }
        catch (RuntimeException e) {
            return null;
        }
    }

	/**
	 * Returns the number of neurons in the hidden layers
	 *
	 * @param neuronType of neuron (use constants in class DefaultNeuron)
	 * @return Number of neurons in the hidden layers
	 */
    public int[] getNumberNeuronsHidden(NeuronType neuronType) {
        int[] hidden = new int[getNeuronLayers().size() - 2];
		for (int x = 1; x < getNeuronLayers().size() - 1; x++)
			hidden[x - 1] = getNeuronLayers().get(x).getNumberOfNeuronsOfType(neuronType);
		return hidden;
    }

    /**
     * Get the number of input neurons in the network.
     * @param neuronType
     * @return
     */
    public int getNumberNeuronsInput(NeuronType neuronType) {
        return getNeuronLayers().get(0).getNumberOfNeuronsOfType(neuronType);
    }

    /**
     * Get the number of output neurons in the network.
     * @param neuronType
     * @return
     */
    public int getNumberNeuronsOutput(NeuronType neuronType) {
        return getNeuronLayers().get(getNumberNeuronsHidden(NeuronType.Normal).length + 1).getNumberOfNeuronsOfType(NeuronType.Normal);
    }

    /**
	 * Feed forwards the input to the output of the input layer
	 *
	 * @param input The vector of inputs to the network.
	 * @return Boolean indicating if the operation has succeeded
	 */
    protected boolean feedForwardInputLayer(Vector<Double> input) {
        try {
            NeuronLayer input_layer = getNeuronLayers().get(0);
            int number_of_bias_nodes = input_layer.getNumberOfNeuronsOfType(NeuronType.Bias);
            
            Vector<Double> padded_input = new Vector<Double>(input);
            for (int i = 0; i < number_of_bias_nodes; i++) {
                padded_input.add(1.0);
            }
            for (int x = 0; x < padded_input.size(); x++) {
                input_layer.getNeuron(x).setInput(padded_input.get(x));
                input_layer.getNeuron(x).calculateOutput();
            }
            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
	}


	/**
	 * Connects neurons from one layer to the next
	 *
	 * @param source Source layer
	 * @param destination Destination layer
	 * @return Boolean indicating if the operation was successful
	 */
	private boolean connectLayers(int source, int destination) {
		try {
            
			int iter2;
			SynapseLayer synapseLayer = new DefaultSynapseLayer();
            Neuron neuron;
                           
			for ( int i = 0; i < getNeuronLayers().get(source).getNetworkSize(); i += 1) {

                neuron = getNeuronLayers().get(source).getNeuron(i);

				// int last_neuron_layer = this.neuron_layers.size() - 1;
				for (iter2 = 0; iter2 < getNeuronLayers().get(destination).getNetworkSize(); iter2++) {
					if (getNeuronLayers().get(destination).getNeuron(iter2).getNeuronType() == NeuronType.Normal) {
						Neuron destination_neuron = getNeuronLayers().get(destination).getNeuron(iter2);
						Synapse synapse = connectNeurons(neuron, destination_neuron);
						synapseLayer.addSynapse(synapse);
					}
				}
			}
			getSynapseLayers().add(synapseLayer);
			return true;
		}
        catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Get the output layer of neurons.
     * @return output layer
     */
    public NeuronLayer getOutputNeurons() {
        return getNeuronLayers().get(getNeuronLayers().size() - 1);
    }

    /**
     * Get an iterator for the output neuron layer vector.
     * @return Iterator
     */
    public Iterator<Neuron> getOutputNeuronLayerIterator () {
        return getOutputNeurons().getNeuronsIterator();
    }

    /**
     * Get a vector of hidden neuron layers
     * @return
     */
    public Vector<NeuronLayer> getHiddenLayers() {
        Vector<NeuronLayer> hiddenLayers = new Vector<NeuronLayer>();
        for (int i = getNeuronLayers().size() - 2; i > 0; i -= 1) {
            hiddenLayers.add(getNeuronLayers().get(i));
        }

        return hiddenLayers;
    }

    /**
     * Get an iterator for the hidden neuron layers vector.
     * @return Iterator
     */
    public Iterator<NeuronLayer> getHiddenNeuronLayersIterator () {
        return getHiddenLayers().iterator();
    }

    /**
     * Reset the values of all neuron layers in the network.
     */
    private void resetValues () {
        for (int i = 0; i < getNeuronLayers().size(); i += 1) {
            getNeuronLayers().get(i).resetValues();
        }
    }

    /**
     * Get all of the neuron layers in the network.
     * @return neuron layer vector
     */
    private Vector<NeuronLayer> getNeuronLayers() {
        return neuronLayers;
    }

    /**
     * Set the neuron layer vector to the provided input.
     * @param neuronLayers
     */
    private void setNeuronLayers(Vector<NeuronLayer> neuronLayers) {
        this.neuronLayers = neuronLayers;
    }

    /**
     * Get an iterator for the neuron layers vector.
     * @return Iterator
     */
    public Iterator<NeuronLayer> getNeuronLayersIterator () {
        return getNeuronLayers().iterator();
    }

    /**
     * Get all of the synapse layers in the network.
     * @return synapse layer vector
     */
    private Vector<SynapseLayer> getSynapseLayers() {
        return synapseLayers;
    }

    /**
     * Get an iterator for the synapse layers vector.
     * @return Iterator
     */
    public Iterator<SynapseLayer> getSynapseLayersIterator () {
        return getSynapseLayers().iterator();
    }

    /**
     * Set the synapse layer vector to the provided input.
     * @param synapseLayers
     */
    private void setSynapseLayers(Vector<SynapseLayer> synapseLayers) {
        this.synapseLayers = synapseLayers;
    }
}
