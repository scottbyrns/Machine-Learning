package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunction;

import java.util.Iterator;
import java.util.Vector;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 5:37:20 PM
 */
public class DefaultNeuron implements Neuron {

    private double input;
    private double output;

    private double delta;

    private NeuronType neuronType;

    private Vector<Synapse> incomingSynapses;
    private Vector<Synapse> outgoingSynapses;

    private ActivationFunction activationFunction;

    public DefaultNeuron(ActivationFunction activationFunction) {
        setActivationFunction(activationFunction);
        setNeuronType(NeuronType.Normal);
        setIncomingSynapses(new Vector<Synapse>());
        setOutgoingSynapses(new Vector<Synapse>());
    }

    public void resetValues () {
        setDelta(0.0);
        setInput(0.0);
        setOutput(0.0);
    }

    public void resetWeights () {
        for (Synapse synapse : outgoingSynapses) {
            synapse.resetWeight();
        }
    }

    public double getInput() {
        return input;
    }

    public void setInput(double input) {
        this.input = input;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public NeuronType getNeuronType() {
        return neuronType;
    }

    public void setNeuronType(NeuronType neuronType) {
        this.neuronType = neuronType;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    private void setIncomingSynapses(Vector<Synapse> incomingSynapses) {
        this.incomingSynapses = incomingSynapses;
    }

    private void setOutgoingSynapses(Vector<Synapse> outgoingSynapses) {
        this.outgoingSynapses = outgoingSynapses;
    }

    public void addIncomingSynapse (Synapse synapse) {
        incomingSynapses.add(synapse);
    }

    /**
     * Get the incoming synapse at a given index.
     * @param index
     * @return The Synapse at that index or null if the index is out of bounds.
     */
    public Synapse getIncomingSynapse (int index) {
        Synapse synapse;
        try {
            synapse = incomingSynapses.get(index);
        }
        catch (ArrayIndexOutOfBoundsException e) {
            synapse = null;
        }
        return synapse;
    }

    public void removeIncomingSynapse (Synapse synapse) {
        incomingSynapses.remove(synapse);
    }

    public void addOutgoingSynapse (Synapse synapse) {
        outgoingSynapses.add(synapse);
    }

    /**
     * Get the outgoing synapse at a given index.
     * @param index
     * @return The Synapse at that index or null if the index is out of bounds.
     */
    public Synapse getOutgoingSynapse (int index) {
        Synapse synapse;
        try {
            synapse = outgoingSynapses.get(index);
        }
        catch (ArrayIndexOutOfBoundsException e) {
            synapse = null;
        }
        return synapse;
    }

    public void removeOutgoingSynapse (Synapse syanpse) {
        outgoingSynapses.remove(syanpse);
    }

    /**
     * Set the activation function of the neuron
     * @param activationFunction
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
     * @return IActivationFunction This neurons activation function
     */
    private ActivationFunction getActivationFunction () {
        return this.activationFunction;
    }

    /**
     * Calculates the neuron's output by activating the current input
     * @return double Output of the neuron.
     */
    public double calculateOutput () {
        setOutput(getActivationFunction().calculate(getInput()));
        return getOutput();
    }

    /**
     * Get an itterator for the outgoing synapse vector.
     * @return
     */
    public Iterator<Synapse> getOutgoingSynapseIterator() {
        return outgoingSynapses.iterator();
    }

	/**
	 * Input the value into the derivative of the activation function and return
	 * the result
	 *
	 * @param value Value to enter into the derivative of the activation function
	 * @return Returns the output of the activation function's derivative after
	 *         inputing the given value
	 */
    public double calculateDerivative(double value) {
        return activationFunction.calculateDerivate(value);
    }

}
