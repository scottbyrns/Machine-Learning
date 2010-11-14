package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.datasets.Pattern;

import java.util.Iterator;
import java.util.Vector;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 1:15:08 PM
 */
public interface FeedForwardNeuralNetwork {

    /**
     * Connect two neurons together and return the connecting synapse.
     * @param source
     * @param destination
     * @return The connecting neuron or null if something went wrong.
     */
    public Synapse connectNeurons(Neuron source, Neuron destination);

    /**
     * Returns the current error for the a vector of patterns
     *
     * @param patterns The list of patterns that will be tested
     * @return The mean squared error for the list of patterns, or -1.0 in case of error
     */
    public double measurePatternListError(Vector<Pattern> patterns);

	/**
	 * Returns the mean squared error between the network's output and the
	 * desired one
	 *
	 * @param output Desired output
	 * @return Returns the mean squared error, or -1.0 in case of error
	 */
	public double meanSquaredError(Vector<Double> output);

	/**
	 * Calculates the network's output by feeding the input all the way to the
	 * output layer
	 *
	 * @param input
	 *            The network's input
	 * @return Returns the network's output, or null in case of error
	 */
	public Vector<Double> feedForward(Vector<Double> input);

	/**
	 * Returns the number of neurons in the hidden layers
	 *
	 * @param neuronType of neuron (use constants in class DefaultNeuron)
	 * @return Number of neurons in the hidden layers
	 */
	public int[] getNumberNeuronsHidden(NeuronType neuronType);

	/**
	 * Returns the number of neurons in the input layer
	 *
	 * @param neuronType of neuron (use constants in class DefaultNeuron)
	 * @return Number of neurons in the input layer
	 */
	public int getNumberNeuronsInput(NeuronType neuronType);


    /**
     * Get the number of output neurons in the network.
     * @param neuronType
     * @return
     */
    public int getNumberNeuronsOutput(NeuronType neuronType);

    /**
     * Get the output layer of neurons.
     * @return output layer
     */
    public NeuronLayer getOutputNeurons();

    /**
     * Get a vector of hidden neuron layers
     * @return
     */
    public Vector<NeuronLayer> getHiddenLayers();

    /**
     * Get all of the synapse layers in the network.
     * @return
     */
    public Vector<SynapseLayer> getSynapseLayers ();

    /**
     * Get all of the neuron layers in the network.
     * @return
     */
    public Vector<NeuronLayer> getNeuronLayers ();

	/**
	 * Returns the mean squared error for a input-output pair
	 *
	 * @param input The input
	 * @param output The desired output
	 * @return The mean squared error for the input-output pair, or -1.0 in case
	 *         of error
	 */
	public double getPredictionError(Vector<Double> input, Vector<Double> output);

	/**
	 * Returns the mean squared error for a set of input-output pairs
	 *
	 * @param patterns A vector of input-output pairs
	 * @return The mean squared error for the set of input-output pairs, or -1.0
	 *         in case of error
	 */
	public double getPredictionError(Vector<Pattern> patterns);

	/**
	 * Gets a vector of weights contained in the neural network
	 *
	 * @return Vector of weight values
	 */
	public Vector<Double> getWeightVector();

    /**
     * Set the weights in the network to the values of the provided vector.
     * 
     * @param weightVector
     */
    public void setWeightVector (Vector<Double> weightVector);

    /**
	 * Returns the predicted pattern for an input
	 *
	 * @param input The input
	 * @return The predicted pattern for the given input, or null in case of
	 *         error
	 */
	public Vector<Double> getPrediction(Vector<Double> input);

    /**
     * Get an iterator for the neuron layers vector.
     * @return Iterator
     */
    public Iterator<NeuronLayer> getNeuronLayersIterator ();

    /**
     * Get an iterator for the synapse layers vector.
     * @return Iterator
     */
    public Iterator<SynapseLayer> getSynapseLayersIterator ();

    /**
     * Get an iterator for the hidden neuron layers vector.
     * @return Iterator
     */
    public Iterator<NeuronLayer> getHiddenNeuronLayersIterator ();

    /**
     * Get an iterator for the output neuron layer vector.
     * @return Iterator
     */
    public Iterator<Neuron> getOutputNeuronLayerIterator ();
}
