package com.scottbyrns.ml.neural;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 9:28:58 AM
 */
public class DefaultSynapse implements Synapse {

    private Neuron inputNeuron;
    private Neuron outputNeuron;
    private double weight;
    private double value;

    public DefaultSynapse(Neuron source, Neuron destination, double weight) {
        setInputNeuron(source);
        setOutputNeuron(destination);
        setWeight(weight);
    }

    public void setInputNeuron(Neuron neuron) {
        inputNeuron = neuron;
    }

    public Neuron getInputNeuron() {
        return inputNeuron;
    }

    public void setOutputNeuron(Neuron neuron) {
        outputNeuron = neuron;
    }

    public Neuron getOutputNeuron() {
        return outputNeuron;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getWeight() {
        return weight;
    }

    public void resetWeight () {
        setWeight(0.0);
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }
}
