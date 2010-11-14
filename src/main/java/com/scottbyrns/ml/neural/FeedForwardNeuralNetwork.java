package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.datasets.Pattern;

import java.util.Iterator;
import java.util.Vector;

/**
 * Feedforward neural network
 * <br />
 * A <strong>feedforward neural network</strong> is an artificial neural network where
 * connections between the units do not form a directed cycle.
 * This is different from recurrent neural networks.
 * <br />
 * The feedforward neural network was the first and arguably simplest type of artificial
 * neural network devised. In this network, the information moves in only one direction,
 * forward, from the input nodes, through the hidden nodes (if any) and to the output nodes.
 * There are no cycles or loops in the network.
 *
 * @author Scott Byrns
 * Date: Nov 11, 2010
 * Time: 1:15:08 PM
 *
 * @version 1.0
 */
public interface FeedForwardNeuralNetwork {

    /**
     * Connect two neurons together and return the connecting synapse.
     * 
     * @param source neuron
     * @param destination neuron
     * @return The connecting neuron or null if something went wrong.
     */
    public Synapse connectNeurons(Neuron source, Neuron destination);

	/**
	 * Calculates the network's output by feeding the input all the way to the
	 * output layer
	 *
	 * @param input The network's input
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
     * 
     * @param neuronType of neuron (use constants in class DefaultNeuron)
     * @return Number of neurons in the output layer
     */
    public int getNumberNeuronsOutput(NeuronType neuronType);
    
    /**
     * Get an iterator for the output neuron layer vector.
     *
     * @return Iterator
     */
    public Iterator<Neuron> getOutputNeuronLayerIterator ();

    /**
     * Get an iterator for the hidden neuron layers vector.
     *
     * @return Iterator
     */
    public Iterator<NeuronLayer> getHiddenNeuronLayersIterator ();

    /**
     * Get an iterator for the neuron layers vector.
     *
     * @return Iterator
     */
    public Iterator<NeuronLayer> getNeuronLayersIterator ();

    /**
     * Get an iterator for the synapse layers vector.
     *
     * @return Iterator
     */
    public Iterator<SynapseLayer> getSynapseLayersIterator ();

    /**
     * Returns the predicted pattern for an input
     *
     * @param input The input
     * @return The predicted pattern for the given input, or null in case of
     *         error
     */
    public Vector<Double> getPrediction(Vector<Double> input);

	/**
	 * Returns the mean squared error for a input-output pair
	 *
	 * @param input The input
	 * @param output The desired output
     *
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
     * @TODO see if this can be deprecated in favor of an iterator
	 */
	public Vector<Double> getWeightVector();

    /**
     * Set the weights in the network to the values of the provided vector.
     *
     * @param weightVector to replace current weight vector
     */
    public void setWeightVector (Vector<Double> weightVector);
}
