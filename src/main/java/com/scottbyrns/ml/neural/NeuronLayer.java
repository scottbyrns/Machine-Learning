package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunction;

import java.util.Iterator;
import java.util.Vector;

/**
 * Representation of a layer of neurons.
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 11:14:39 AM
 */
public interface NeuronLayer {

    /**
     * Calculate the neurons output by passing inputs by the activation function.
     *
     * @return Boolean indication of operations success.
     */
    public boolean calculateOutput ();

    /**
     * Forward the values to the next layer.
     *
     * @return Boolean indication of operations success.
     */
    public boolean feedForward ();

    /**
     * Get the neuron layers input
     *
     * @return a vector of all the neurons inputs.
     */
    public Vector<Double> getInput ();
    /**
     * Get the neuron layers output
     *
     * @return a vector of all the neurons outputs.
     */
    public Vector<Double> getOutput ();

    /**
     * Reset the layers neurons values.
     *
     * @return Boolean indication of operations success.
     */
    public boolean resetValues ();

    /**
     * Reset the layers neurons weights.
     *
     * @return Boolean indication of operations success.
     */
    public boolean resetWeights ();

    /**
     * Get the neuron at the index specified.
     *
     * @param index
     * @return
     *
     * @TODO remove this method from the public api
     */
    public Neuron getNeuron (int index);

    /**
     * Get the count of neurons of the specified type.
     * 
     * @param type of neuron to count.
     * @return A count of neurons matching the specified type.
     */
    public int getNumberOfNeuronsOfType (NeuronType type);

    /**
     * Set the activation function of the whole neural layer
     *
     * @param activationFunction activation function to use for all neurons in this layer
     * @return Boolean indication of operations success.
     */
    public boolean setActivationFunction (ActivationFunction activationFunction);

    /**
     * Get an iterator for the neuron vector.
     * 
     * @return Iterator
     */
    public Iterator<Neuron> getNeuronsIterator ();
}
