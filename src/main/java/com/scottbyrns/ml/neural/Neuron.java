package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunction;

import java.io.Serializable;
import java.util.Iterator;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 5:33:08 PM
 */
public interface Neuron extends Serializable {

    /**
     * Get the input of the neuron.
     * @return
     */
    public double getInput ();

    /**
     * Set the input of the neuron.
     * @param input
     */
    public void setInput (double input);

    /**
     * Get the neurons output value.
     * @return
     */
    public double getOutput ();

    /**
     * Set the neurons output value.
     * @param output
     */
    public void setOutput (double output);

    /**
     * Get the Neurons delta
     * @return
     */
    public double getDelta ();

    /**
     * Set the Neurons delta.
     * @param delta
     */
    public void setDelta (double delta);

    /**
     * Reset the input, output, and delta values of the neuron.
     */
    public void resetValues ();

    /**
     * Reset the weights of all outgoing synapses.
     */
    public void resetWeights ();

    /**
     * Get the neuron type.
     * @return
     */
    public NeuronType getNeuronType ();

    /**
     * Set the neuron type.
     * @param neuronType
     */
    public void setNeuronType (NeuronType neuronType);

    /**
     * Add an incoming synapse to the neuron.
     * @param synapse
     */
    public void addIncomingSynapse (Synapse synapse);

    /**
     * Get the incoming synapse at a given index.
     * @TODO make non index lookup possible. (Iterator)
     * @param index
     * @return
     */
    public Synapse getIncomingSynapse (int index);

    /**
     * Remove an incoming synapse.
     * @param synapse
     */
    public void removeIncomingSynapse (Synapse synapse);

    /**
     * Add an outgoing synapse to the neuron.
     * @param synapse
     */
    public void addOutgoingSynapse (Synapse synapse);

    /**
     * Get the outgoing synapse at a given index. 
     * @param index
     * @return
     */
    public Synapse getOutgoingSynapse (int index);

    /**
     * Get an iterator for the outgoing synapse vector.
     * @return
     */
    public Iterator<Synapse> getOutgoingSynapseIterator();

    /**
     * Remove an outgoing synapse.
     * @param synapse
     */
    public void removeOutgoingSynapse (Synapse synapse);


    /**
     * Set the activation function of the neuron
     * @param activationFunction
     * @return Boolean indication of operations success.
     */
    public boolean setActivationFunction (ActivationFunction activationFunction);

    /**
     * Calculates the neuron's output by activating the current input
     * @return double Output of the neuron.
     */
    public double calculateOutput ();

	/**
	 * Input the value into the derivative of the activation function and return
	 * the result
	 *
	 * @param value Value to enter into the derivative of the activation function
	 * @return Returns the output of the activation function's derivative after
	 *         inputing the given value
	 */
    public double calculateDerivative(double value);
}
