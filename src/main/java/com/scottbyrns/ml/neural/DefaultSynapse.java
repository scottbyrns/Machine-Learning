package com.scottbyrns.ml.neural;

/**
 * An artificial synapse that is an abstraction of biological
 * synapses in the form of a weight unit.
 * 
 * @author Scott Byrns
 * Date: Nov 11, 2010
 * Time: 9:28:58 AM
 *
 * @version 1.0
 */
public class DefaultSynapse implements Synapse {

    private Neuron inputNeuron, outputNeuron;
    private double weight, value;


    /**
     * Create a new synapse that spans the source and destination neurons provided
     * with the specified weight.
     *
     * @param source neuron
     * @param destination neuron
     * @param weight Weight of the neuron
     */
    public DefaultSynapse(Neuron source, Neuron destination, double weight) {
        setInputNeuron(source);
        setOutputNeuron(destination);
        setWeight(weight);
    }

    /**
     * Set the input neuron of the synapse.
     *
     * @param neuron input neuron of the synapse
     */
    public void setInputNeuron(Neuron neuron) {
        inputNeuron = neuron;
    }

    /**
     * Get the input neuron of the synapse.
     *
     * @return input neuron of the synapse
     */
    public Neuron getInputNeuron() {
        return inputNeuron;
    }

    /**
     * Set the output neuron of the synapse.
     *
     * @param neuron output neuron of the synapse
     */
    public void setOutputNeuron(Neuron neuron) {
        outputNeuron = neuron;
    }

    /**
     * Get the output neuron of the synapse.
     *
     * @return output neuron of the synapse
     */
    public Neuron getOutputNeuron() {
        return outputNeuron;
    }

    /**
     * Get the weight of the synapse.
     *
     * @return weight of the synapse
     */
    public double getWeight() {
        return weight;
    }

    /**
     * Set the weight of the synapse.
     *
     * @param weight of the synapse
     */
    public void setWeight(double weight) {
        this.weight = weight;
    }

    /**
     * Reset the weight of the synapse.
     */
    public void resetWeight () {
        setWeight(0.0);
    }

    /**
     * Get the value of the synapse.
     *
     * @return value of the synapse
     */
    public double getValue() {
        return value;
    }

    /**
     * Set the value of the synapse.
     *
     * @param value of the synapse
     */
    public void setValue(double value) {
        this.value = value;
    }
}
