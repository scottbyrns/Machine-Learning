package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunctionLinear;
import com.scottbyrns.ml.neural.Activation.ActivationFunctionSigmoid;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.*;

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

        Iterator<Neuron> neuronIterator = neuralLayer.getNeuronsIterator();

        while (neuronIterator.hasNext()) {
            neuronIterator.next().setInput(0.123);
        }

        neuralLayer.calculateOutput();
        List<Double> output = neuralLayer.getOutput();

        Iterator<Double> outputIterator = output.iterator();

        while (outputIterator.hasNext()) {
            assertEquals(0.123, outputIterator.next(), 0.001);
        }

    }

    @Test
    public void testFeedForward () {

        Iterator<Neuron> neuronIterator = neuralLayer.getNeuronsIterator();

        Neuron neuron;
        Synapse synapse;

        while(neuronIterator.hasNext()) {
            neuron = neuronIterator.next();

            synapse = new DefaultSynapse(neuron, neuron, 0.1);
            synapse.setValue(1);
            neuron.addIncomingSynapse(synapse);

            synapse = new DefaultSynapse(neuron, neuron, 0.1);
            synapse.setValue(1);
            neuron.addOutgoingSynapse(synapse);
            neuron.setInput(1);
            
        }

        neuralLayer.feedForward();

        neuronIterator = neuralLayer.getNeuronsIterator();

        while(neuronIterator.hasNext()) {
            neuron = neuronIterator.next();
            neuron.calculateOutput();
            assertEquals(1.1, neuron.getOutput(), 0.001);
        }

    }

    @Test
    public void testGetOutput () {

        List<Double> neuronOutputVector = neuralLayer.getOutput();

        Iterator<Double> neuronOutputVectorIterator = neuronOutputVector.iterator();

        while (neuronOutputVectorIterator.hasNext()) {
            assertEquals(0.0, neuronOutputVectorIterator.next(), 0.001);
        }
    }

    @Test
    public void testResetValues () {

        Iterator<Neuron> neuronIterator = neuralLayer.getNeuronsIterator();
        Neuron neuron;

        while (neuronIterator.hasNext()) {
            neuron = neuronIterator.next();
            neuron.setInput(1.234);
            neuron.setDelta(1.234);
            neuron.setOutput(1.234);
        }

        neuralLayer.resetValues();

        neuronIterator = neuralLayer.getNeuronsIterator();


        while (neuronIterator.hasNext()) {
            neuron = neuronIterator.next();

            assertNotSame(1.234, neuron.getInput());
            assertEquals(0.0, neuron.getInput(), 0.001);

            assertNotSame(1.234, neuron.getDelta());
            assertEquals(0.0, neuron.getDelta(), 0.001);

            assertNotSame(1.234, neuron.getOutput());
            assertEquals(0.0, neuron.getOutput(), 0.001);
        }
    }

    @Test
    public void testResetWeights () {

        List<Synapse> synapses = new ArrayList<Synapse>();
        Synapse synapse;

        for (int i = 0; i < 10; i += 1) {
            synapse = new DefaultSynapse(neuralLayer.getNeuronsIterator().next(), neuralLayer.getNeuronsIterator().next(), 0.1);
            synapse.setWeight(0.123);
            synapses.add(synapse);
            neuralLayer.getNeuronsIterator().next().addOutgoingSynapse(synapse);
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
        neuralLayer.getNeuronsIterator().next().setInput(1.0);
        neuralLayer.setActivationFunction(new ActivationFunctionSigmoid());
        double output = neuralLayer.getNeuronsIterator().next().calculateOutput();
        assertEquals(0.7310585786300049, output, 0.000000000001);
    }

    @Test
    public void testGetNeuronsIterator () {
        Iterator<Neuron> neuronIterator = neuralLayer.getNeuronsIterator();

        assertTrue(neuronIterator.hasNext());
    }

    @After
    public void teardown () {
        neuralLayer = null;
    }
}
