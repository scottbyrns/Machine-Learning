package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunctionLinear;
import com.scottbyrns.ml.neural.Activation.ActivationFunctionSigmoid;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 11:48:02 AM
 */
public class NeuronLayerTest {

    private NeuronLayer neuralLayer;

    @Before
    public void setup () {
        neuralLayer = new DefaultNeuronLayer(5, new ActivationFunctionLinear(), 1);
    }

    @Test
    public void testOverloadedConstructor () {
        neuralLayer = new DefaultNeuronLayer(5, new ActivationFunctionLinear());

        int count = 0;

        Iterator<Neuron> neuronIterator = neuralLayer.getNeuronsIterator();

        while (neuronIterator.hasNext()) {
            neuronIterator.next();
            count += 1;
        }

        assertEquals(5, count);
    }

    @Test
    public void testCalculateOutput () {
        neuralLayer.calculateOutput();
        Vector<Double> output = neuralLayer.getOutput();
        /**
         * @TODO Finish this test.
         */
    }

    @Test
    public void testFeedForward () {
        /**
         * @TODO Figure out a test for feedforward.
         */
    }

    @Test
    public void testGetInput () {
        Vector<Double> input = neuralLayer.getInput();
        Iterator<Double> inputIterable = input.iterator();

        double inputValue;

        while (inputIterable.hasNext()) {
            inputValue = inputIterable.next();
            assertEquals(0.0, inputValue, 0.001);
        }
    }

    @Test
    public void testGetOutput () {

    }

    @Test
    public void testResetValues () {
        neuralLayer.getNeuron(0).setInput(1.234);
        neuralLayer.resetValues();
        assertNotSame(1.234, neuralLayer.getNeuron(0).getInput());
        assertEquals(0.0, neuralLayer.getNeuron(0).getInput(), 0.001);
    }

    @Test
    public void testResetWeights () {

        List<Synapse> synapses = new ArrayList<Synapse>();
        Synapse synapse;

        for (int i = 0; i < 10; i += 1) {
            synapse = new DefaultSynapse(neuralLayer.getNeuron(0), neuralLayer.getNeuron(0), 0.1);
            synapse.setWeight(0.123);
            synapses.add(synapse);
            neuralLayer.getNeuron(0).addOutgoingSynapse(synapse);
            /* Just in case this test runs before testGetWeight / testSetWeight */
            assertEquals(0.123, synapse.getWeight(), 0.001);
        }

        neuralLayer.resetWeights();

        for (int i = 0; i < 10; i += 1) {
            synapse = synapses.get(i);
            assertEquals(0.0, synapse.getWeight(), 0.001);
        }

    }

    @Test
    public void testGetNumberOfNeuronsOfTypeNormal () {
        assertEquals(5, neuralLayer.getNumberOfNeuronsOfType(NeuronType.Normal));
    }

    @Test
    public void testGetNumberOfNeuronsOfTypeBias () {
        assertEquals(1, neuralLayer.getNumberOfNeuronsOfType(NeuronType.Bias));
    }

    @Test
    public void testSetActivationFunction () {
        neuralLayer.getNeuron(0).setInput(1.0);
        neuralLayer.setActivationFunction(new ActivationFunctionSigmoid());
        double output = neuralLayer.getNeuron(0).calculateOutput();
        assertEquals(0.7310585786300049, output, 0.000000000001);
    }

    @After
    public void teardown () {
        neuralLayer = null;
    }
}
