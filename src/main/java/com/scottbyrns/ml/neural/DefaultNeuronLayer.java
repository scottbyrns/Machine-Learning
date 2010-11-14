package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunction;

import java.util.Iterator;
import java.util.Vector;

/**
 * Representation of a layer of neurons.
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 11:37:24 AM
 */
public class DefaultNeuronLayer implements NeuronLayer {

    Vector<Neuron> neurons;

    public DefaultNeuronLayer(int networkSize, ActivationFunction activationFunction) {
        this(networkSize, activationFunction, 0);
    }

    public DefaultNeuronLayer(int networkSize, ActivationFunction activationFunction, int biasCount) {
        setNeurons(new Vector<Neuron>());
        Neuron neuron;
        for (int i = 0; i < networkSize; i += 1) {

            neuron = new DefaultNeuron(activationFunction);
            neuron.setNeuronType(NeuronType.Normal);

            getNeurons().add(neuron);
        }

        for (int i = 0; i < biasCount; i += 1) {
            neuron = new DefaultNeuron(activationFunction);
            neuron.setNeuronType(NeuronType.Bias);

            getNeurons().add(neuron);
        }
    }

    /**
     * Get the size of this layers network of neurons.
     * @return
     */
    public int getNetworkSize () {
        return getNeurons().size();
    }

    /**
     * Calculate the neurons output by passing inputs by the activation function.
     * @return Boolean indication of operations success.
     */
    public boolean calculateOutput () {
        try {
            Iterator<Neuron> neuronIterator = getNeuronsIterator();
            Neuron neuron;
            while(neuronIterator.hasNext()) {
                neuron = neuronIterator.next();
                neuron.calculateOutput();
            }
            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Forward the values to the next layer.
     * @return
     */
    public boolean feedForward() {
        try {
            Iterator<Neuron> neuronIterator = getNeuronsIterator();
            Iterator<Synapse> synapseItterator;

            Neuron neuron;
            Neuron destinationNeuron;
            double destinationInput;

            Synapse synapse;

            while(neuronIterator.hasNext()) {
                neuron = neuronIterator.next();
                synapseItterator = neuron.getOutgoingSynapseIterator();
                while(synapseItterator.hasNext()) {

                    synapse = synapseItterator.next();
                    destinationNeuron = synapse.getOutputNeuron();
                    destinationInput = destinationNeuron.getInput();

                    synapse.setValue(neuron.getOutput());
                    destinationInput += synapse.getValue() * synapse.getWeight();
                    destinationNeuron.setInput(destinationInput);
                }
            }

            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }


    /**
     * Get the neuron layers input
     * @return a vector of all the neurons inputs.
     */
    public Vector<Double> getInput() {
        Vector<Double> input = new Vector<Double>();

        Iterator<Neuron> neuronIterator = getNeuronsIterator();

        while(neuronIterator.hasNext()) {
            input.add(neuronIterator.next().getInput());
        }

        return input;
    }

    /**
     * Get the neuron layers output
     * @return a vector of all the neurons outputs.
     */
    public Vector<Double> getOutput() {

        Vector<Double> output = new Vector<Double>();

        Iterator<Neuron> neuronIterator = getNeuronsIterator();

        while (neuronIterator.hasNext()) {
            output.add(neuronIterator.next().getOutput());
        }

        return output;
    }

    /**
     * Reset the layers neurons values.
     * @return Boolean indication of operations success.
     */
    public boolean resetValues() {
        try {
            Iterator<Neuron> neuronIterator = getNeuronsIterator();
            while (neuronIterator.hasNext()) {
                neuronIterator.next().resetValues();
            }
            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Reset the layers neurons weights.
     * @return Boolean indication of operations success.
     */
    public boolean resetWeights() {
        try {
            Iterator<Neuron> neuronIterator = getNeuronsIterator();
            while (neuronIterator.hasNext()) {
                neuronIterator.next().resetWeights();
            }
            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Get the neuron at the index specified.
     * @param index
     * @return
     */
    public Neuron getNeuron(int index) {
        try {
            return getNeurons().get(index);
        }
        catch (NullPointerException e) {
            return null;
        }
    }

    /**
     * Get the count of neurons of the specified type.
     * @param type
     * @return
     */
    public int getNumberOfNeuronsOfType(NeuronType type) {
        try {
            Iterator<Neuron> neuronIterator = getNeuronsIterator();

            int count = 0;

            while (neuronIterator.hasNext()) {
                if (neuronIterator.next().getNeuronType() == type) {
                    count += 1;
                }
            }
            return count;
        }
        catch (RuntimeException e) {
            return 0;
        }
    }


    /**
     * Set the activation function of the whole neural layer
     * @param activationFunction
     * @return Boolean indication of operations success.
     */
    public boolean setActivationFunction(ActivationFunction activationFunction) {
        try {
            Iterator<Neuron> neuronIterator = getNeuronsIterator();
            while (neuronIterator.hasNext()) {
                neuronIterator.next().setActivationFunction(activationFunction);
            }
            return true;
        }
        catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Get the neuron vector of this layer
     * @return neuron vector
     */
    private Vector<Neuron> getNeurons () {
        return neurons;
    }

    /**
     * Set the neurons vector to the provided input
     * @param neurons new neuron vector
     */
    private void setNeurons (Vector<Neuron> neurons) {
        this.neurons = neurons;
    }

    /**
     * Get an iterator for the neuron vector.
     * @return Iterator
     */
    public Iterator<Neuron> getNeuronsIterator () {
        return getNeurons().iterator();
    }
}
