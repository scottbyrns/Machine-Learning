package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunction;

import java.util.Iterator;
import java.util.Vector;

/**
 * An artificial neuron that is an abstraction of biological
 * neurons in the form of a linear threshold model.
 *
 * @author Scott Byrns
 * Date: Nov 10, 2010
 * Time: 5:37:20 PM
 *
 * @version 1.0
 */
public class DefaultNeuron implements Neuron {

    private double input, output, delta;
    
    private Vector<Synapse> incomingSynapses, outgoingSynapses;
    private ActivationFunction activationFunction;
    private NeuronType neuronType;

    public DefaultNeuron(ActivationFunction activationFunction) {
        setActivationFunction(activationFunction);
        setNeuronType(NeuronType.Normal);
        setIncomingSynapses(new Vector<Synapse>());
        setOutgoingSynapses(new Vector<Synapse>());
    }

    /**
     * Get the input of the neuron.
     *
     * @return input of the neuron
     */
    public double getInput() {
        return input;
    }

    /**
     * Set the input of the neuron.
     *
     * @param input of the neuron
     */
    public void setInput(double input) {
        this.input = input;
    }

    /**
     * Get the neurons output value.
     *
     * @return neurons output value
     */
    public double getOutput() {
        return output;
    }

    /**
     * Set the neurons output value.
     *
     * @param output value of the neuron
     */
    public void setOutput(double output) {
        this.output = output;
    }

    /**
     * Get the Neurons delta
     *
     * @return neurons delta
     */
    public double getDelta() {
        return delta;
    }

    /**
     * Set the Neurons delta.
     *
     * @param delta of the neuron
     */
    public void setDelta(double delta) {
        this.delta = delta;
    }

    /**
     * Reset the input, output, and delta values of the neuron.
     */
    public void resetValues () {
        setDelta(0.0);
        setInput(0.0);
        setOutput(0.0);
    }

    /**
     * Reset the weights of all outgoing synapses.
     */
    public void resetWeights () {
        for (Synapse synapse : outgoingSynapses) {
            synapse.resetWeight();
        }
    }

    /**
     * Get the neuron type.
     *
     * @return type of the neuron
     */
    public NeuronType getNeuronType() {
        return neuronType;
    }

    /**
     * Set the neuron type.
     *
     * @param neuronType of the neuron
     */
    public void setNeuronType(NeuronType neuronType) {
        this.neuronType = neuronType;
    }

    /**
     * Add an incoming synapse to the neuron.
     *
     * @param synapse incoming synapse
     */
    public void addIncomingSynapse (Synapse synapse) {
        incomingSynapses.add(synapse);
    }

    /**
     * Set the incomingSynapses vector to the provided vector.
     * 
     * @param incomingSynapses vector
     */
    private void setIncomingSynapses(Vector<Synapse> incomingSynapses) {
        this.incomingSynapses = incomingSynapses;
    }

    /**
     * Get an iterator for the incoming synapse vector.
     *
     * @return
     */
    public Iterator<Synapse> getIncomingSynapseIterator() {
        return incomingSynapses.iterator();
    }

    /**
     * Remove a synapse from the incoming synapse vector.
     *
     * @param synapse to remove
     */
    public void removeIncomingSynapse (Synapse synapse) {
        incomingSynapses.remove(synapse);
    }

    /**
     * Add an outgoing synapse to the neuron.
     *
     * @param synapse outgoing synapse
     */
    public void addOutgoingSynapse (Synapse synapse) {
        outgoingSynapses.add(synapse);
    }

    /**
     * Set the outgoingSynapses vector to the provided vector.
     * 
     * @param outgoingSynapses vector
     */
    private void setOutgoingSynapses(Vector<Synapse> outgoingSynapses) {
        this.outgoingSynapses = outgoingSynapses;
    }

    /**
     * Get an iterator for the outgoing synapse vector.
     *
     * @return iterator for the outgoing synapse vector
     */
    public Iterator<Synapse> getOutgoingSynapseIterator() {
        return outgoingSynapses.iterator();
    }

    /**
     * Remove an outgoing synapse.
     *
     * @param synapse to remove
     */
    public void removeOutgoingSynapse (Synapse synapse) {
        outgoingSynapses.remove(synapse);
    }

    /**
     * Set the activation function of the neuron
     *
     * @param activationFunction for this neuron
     * @return Boolean indication of operations success.
     */
    public boolean setActivationFunction (ActivationFunction activationFunction) {
        if (null != activationFunction) {
            this.activationFunction = activationFunction;
            return true;
        }
        return false;
    }

    /**
     * Get the activation function of the neuron
     * 
     * @return IActivationFunction This neurons activation function
     */
    private ActivationFunction getActivationFunction () {
        return this.activationFunction;
    }

    /**
     * Calculates the neuron's output by activating the current input
     *
     * @return double Output of the neuron.
     */
    public double calculateOutput () {
        setOutput(getActivationFunction().calculate(getInput()));
        return getOutput();
    }

	/**
	 * Input the value into the derivative of the activation function and return
	 * the result
	 *
	 * @param value Value to enter into the derivative of the activation function
	 * @return Returns the output of the activation function's derivative after
	 *         supplying the given value
	 */
    public double calculateDerivative(double value) {
        return activationFunction.calculateDerivative(value);
    }

}
