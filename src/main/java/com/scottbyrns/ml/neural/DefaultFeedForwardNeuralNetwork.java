package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.Mathematics;
import com.scottbyrns.ml.datasets.Pattern;
import com.scottbyrns.ml.neural.Activation.ActivationFunction;
import com.scottbyrns.ml.neural.Activation.ActivationFunctionSigmoid;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Default implementation of the FeedForwardNeuralNetwork interface.
 *
 * @author Scott Byrns
 * Date: Nov 11, 2010
 * Time: 2:10:48 PM
 *
 * @version 1.0
 */
public class DefaultFeedForwardNeuralNetwork implements FeedForwardNeuralNetwork {

    public static final int DEFAULT_NUMBER_BIAS_NEURONS = 1;

    private List<NeuronLayer> neuronLayers;
    private List<SynapseLayer> synapseLayers;

    /**
     * Create an instance of a DefaultFeedForwardNeuralNetwork from an existing FeedForwardNeuralNetwork
     *
     * @param feedForwardNeuralNetwork neural network to copy.
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
     *
     * @param inputSize number of inputs
     * @param hiddenSizes array of hidden layer neuron counts
     * @param outputSize number of outputs
     */
    public DefaultFeedForwardNeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize) {
        this(inputSize, hiddenSizes, outputSize, new ActivationFunctionSigmoid());
    }

    /**
     * Create a new DefaultFeedForwardNeuralNetwork
     *
     * @param inputSize number of inputs
     * @param hiddenSizes array of hidden layer neuron counts
     * @param outputSize number of outputs
     * @param activationFunction neuron activation function.
     */
    public DefaultFeedForwardNeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, ActivationFunction activationFunction) {
        try {

            // Create the layers
			int bias_nodes_per_layer = DefaultFeedForwardNeuralNetwork.DEFAULT_NUMBER_BIAS_NEURONS;
			setNeuronLayers(new ArrayList<NeuronLayer>());
			getNeuronLayers().add(new DefaultNeuronLayer(inputSize, activationFunction, bias_nodes_per_layer));

            for (int x = 0; x < hiddenSizes.length; x++) {
                getNeuronLayers().add(new DefaultNeuronLayer(hiddenSizes[x], activationFunction, bias_nodes_per_layer));
            }

			getNeuronLayers().add(new DefaultNeuronLayer(outputSize, activationFunction));
			// Connect the layers
			setSynapseLayers(new ArrayList<SynapseLayer>());
			for (int x = 0; x < getNeuronLayers().size() - 1; x++) {
                connectLayers(x, x + 1);
            }

        }
        catch (RuntimeException e) {
            e.printStackTrace();
        }
    }

    /**
     * Connect two neurons together and return the connecting synapse.
     *
     * @param source neuron
     * @param destination neuron
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
     * Calculates the network's output by feeding the input all the way to the
     * output layer
     *
     * @param input The network's input
     * @return Returns the network's output, or null in case of error
     */
    public List<Double> feedForward(List<Double> input) {
        try {
            resetValues();
            feedForwardInputLayer(input);

            Iterator<NeuronLayer> neuronIterator = getNeuronLayersIterator();
            NeuronLayer neuronLayer;

            neuronIterator.next().feedForward();

            while (neuronIterator.hasNext()) {
                neuronLayer = neuronIterator.next();

                if (neuronIterator.hasNext()) {
                    neuronLayer.calculateOutput();
                    neuronLayer.feedForward();
                }
                else {
                    neuronLayer.calculateOutput();
                    return neuronLayer.getOutput();
                }
            }

            return null;
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
        for (int x = 1; x < getNeuronLayers().size() - 1; x++) {
            hidden[x - 1] = getNeuronLayers().get(x).getNumberOfNeuronsOfType(neuronType);
        }

        return hidden;
    }

	/**
	 * Returns the number of neurons in the input layer matching the NeuronType specified.
	 *
	 * @param neuronType of neuron (use constants in class DefaultNeuron)
	 * @return Number of neurons in the input layer
	 */
    public int getNumberNeuronsInput(NeuronType neuronType) {
        return getNeuronLayers().get(0).getNumberOfNeuronsOfType(neuronType);
    }

    /**
     * Get the number of output neurons in the network matching the NeuronType specified.
     *
     * @param neuronType of neuron (use constants in class DefaultNeuron)
     * @return Number of neurons in the output layer
     */
    public int getNumberNeuronsOutput(NeuronType neuronType) {
        return getNeuronLayers().get(getNumberNeuronsHidden(NeuronType.Normal).length + 1).getNumberOfNeuronsOfType(NeuronType.Normal);
    }

    /**
     * Get the output layer of neurons.
     *
     * @return output layer
     */
    private NeuronLayer getOutputNeurons() {
        return getNeuronLayers().get(getNeuronLayers().size() - 1);
    }

    /**
     * Get an iterator for the output neuron layer vector.
     *
     * @return Iterator
     */
    public Iterator<Neuron> getOutputNeuronLayerIterator () {
        return getOutputNeurons().getNeuronsIterator();
    }

    /**
     * Get an iterator for the hidden neuron layers vector.
     *
     * @return Iterator
     */
    public Iterator<NeuronLayer> getHiddenNeuronLayersIterator () {
        return getHiddenLayers().iterator();
    }

    /**
     * Get an iterator for the neuron layers vector.
     *
     * @return Iterator
     */
    public Iterator<NeuronLayer> getNeuronLayersIterator () {
        return getNeuronLayers().iterator();
    }

    /**
     * Get an iterator for the synapse layers vector.
     *
     * @return Iterator
     */
    public Iterator<SynapseLayer> getSynapseLayersIterator () {
        return getSynapseLayers().iterator();
    }

    /**
     * Returns the predicted pattern for an input
     *
     * @param input The input
     * @return The predicted pattern for the given input, or null in case of
     *         error
     */
    public List<Double> getPrediction(List<Double> input) {
        feedForward(input);
        return getOutput();
    }

    /**
     * Returns the mean squared error for a input-output pair
     *
     * @param input The input
     * @param output The desired output
     *
     * @return The mean squared error for the input-output pair, or -1.0 in case
     *         of error
     */
    public double getPredictionError(List<Double> input, List<Double> output) {
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
    public double getPredictionError(List<Pattern> patterns) {
        try {
            double error = 0;

            Iterator<Pattern> patternIterator = patterns.iterator();
            Pattern pattern;

            while (patternIterator.hasNext()) {
                pattern = patternIterator.next();

                error += getPredictionError(pattern.getInput(), pattern.getOutput());
            }

            return error / patterns.size();

        }
        catch (RuntimeException e) {
            return -1.0;
        }
    }

    /**
     * Gets a vector of weights contained in the neural network
     *
     * @return ArrayList of weight values
     * @TODO see if this can be deprecated in favor of an iterator
     */
    public List<Double> getWeightVector() {
        List<Double> list = new ArrayList<Double>();

        Iterator<SynapseLayer> synapseLayerIterator = getSynapseLayersIterator();
        Iterator<Double> doubleIterator;

        while (synapseLayerIterator.hasNext()) {
            doubleIterator = synapseLayerIterator.next().getWeightVector().iterator();
            
            while (doubleIterator.hasNext()) {
                list.add(doubleIterator.next());
            }
        }

        return list;
    }

    /**
     * Set the weights in the network to the values of the provided vector.
     *
     * @param weightArrayList to replace current weight vector
     */
    public void setWeightVector (List<Double> weightArrayList) {

        Iterator<SynapseLayer> synapseLayerIterator = getSynapseLayers().iterator();
        Iterator<Double> weightArrayListIterator = weightArrayList.iterator();

        while (synapseLayerIterator.hasNext()) {
            synapseLayerIterator.next().setWeightVector(weightArrayListIterator);
        }
    }


	/**
	 * Returns the network's output
	 *
	 * @return Returns an array of doubles with the network's output
	 */
	private List<Double> getOutput() {
		List<Double> output = new ArrayList<Double>();

        Iterator<Neuron> outputNeuronLayerIterator = getOutputNeuronLayerIterator();

        while (outputNeuronLayerIterator.hasNext()) {
            output.add(outputNeuronLayerIterator.next().getOutput());
        }

		return output;
	}











    /**
     * Returns the current error for the a vector of patterns
     *
     * @note Nov 14, 2010 Even though this method is not in use please retain it for future use.
     *
     * @param patterns The list of patterns that will be tested
     * @return The mean squared error for the list of patterns, or -1.0 in case of error
     */
    private double measurePatternListError(List<Pattern> patterns) {
        try {
            double error = 0;
			int count = 0;
            /**
             * @TODO use an iterator
             */
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
    private double meanSquaredError(List<Double> output) {
        try {
			double sum = 0;

			int lastLayerIndex = getNeuronLayers().size() - 1;
			NeuronLayer lastLayer = getNeuronLayers().get(lastLayerIndex);

            Iterator<Neuron> neuronIterator = lastLayer.getNeuronsIterator();
            Iterator<Double> outputIterator = output.iterator();

            while (neuronIterator.hasNext()) {
                double networkOutput = neuronIterator.next().getOutput();
                double desiredOutput = outputIterator.next();

                sum += Math.pow(desiredOutput - networkOutput, 2);
            }

            return sum / 2;

        }
        catch (RuntimeException e) {
            return -1.0;
        }
    }

    /**
	 * Feed forwards the input to the output of the input layer
	 *
	 * @param input The vector of inputs to the network.
	 * @return Boolean indicating if the operation has succeeded
	 */
    protected boolean feedForwardInputLayer(List<Double> input) {
        try {

            NeuronLayer inputLayer = getNeuronLayersIterator().next();

            int numberOfBiasNodes = inputLayer.getNumberOfNeuronsOfType(NeuronType.Bias);

            List<Double> paddedInput = new ArrayList<Double>(input);
            for (int i = 0; i < numberOfBiasNodes; i += 1) {
                paddedInput.add(1.0);
            }

            Iterator<Double> paddedInputIterator = paddedInput.iterator();
            Iterator<Neuron> inputLayerIterator = inputLayer.getNeuronsIterator();

            Neuron inputNeuron;

            while(paddedInputIterator.hasNext() && inputLayerIterator.hasNext()) {
                inputNeuron = inputLayerIterator.next();
                inputNeuron.setInput(paddedInputIterator.next());
                inputNeuron.calculateOutput();
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
            Neuron neuron;
            Neuron destinationNeuron;

            SynapseLayer synapseLayer = new DefaultSynapseLayer();
            Synapse synapse;

            Iterator<Neuron> neuronSourceIterator = getNeuronLayers().get(source).getNeuronsIterator();
            Iterator<Neuron> neuronDestinationIterator;

            while (neuronSourceIterator.hasNext()) {
                neuron = neuronSourceIterator.next();
                neuronDestinationIterator = getNeuronLayers().get(destination).getNeuronsIterator();

                while (neuronDestinationIterator.hasNext()) {
                    destinationNeuron = neuronDestinationIterator.next();

                    if (destinationNeuron.getNeuronType() == NeuronType.Normal) {
                        synapse = connectNeurons(neuron, destinationNeuron);
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
     * Get a vector of hidden neuron layers
     *
     * @return hidden neurons layers
     */
    private List<NeuronLayer> getHiddenLayers() {
        List<NeuronLayer> hiddenLayers = new ArrayList<NeuronLayer>();
        /**
         * @TODO use an iterator
         */
        for (int i = getNeuronLayers().size() - 2; i > 0; i -= 1) {
            hiddenLayers.add(getNeuronLayers().get(i));
        }

        return hiddenLayers;
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
     *
     * @return neuron layer vector
     */
    private List<NeuronLayer> getNeuronLayers() {
        return neuronLayers;
    }

    /**
     * Set the neuron layer vector to the provided input.
     *
     * @param neuronLayers vector to set as current neuron layers
     */
    private void setNeuronLayers(List<NeuronLayer> neuronLayers) {
        this.neuronLayers = neuronLayers;
    }

    /**
     * Get all of the synapse layers in the network.
     *
     * @return synapse layer vector
     */
    private List<SynapseLayer> getSynapseLayers() {
        return synapseLayers;
    }

    /**
     * Set the synapse layer vector to the provided input.
     *
     * @param synapseLayers vector to set as current synapse layers.
     */
    private void setSynapseLayers(List<SynapseLayer> synapseLayers) {
        this.synapseLayers = synapseLayers;
    }
}
