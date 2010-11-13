package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunctionLinear;
import com.scottbyrns.ml.neural.Activation.ActivationFunctionSigmoid;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 9:02:00 AM
 */
public class NeuronTest {

    private Neuron neuron;

    @Before
    public void setup () {
        neuron = new DefaultNeuron(new ActivationFunctionLinear());
    }

    @Test
    public void testGetInput () {
        assertEquals(0, neuron.getInput(), 0.001);
    }

    @Test
    public void testSetInput () {
        neuron.setInput(0.1);
        assertEquals(0.1, neuron.getInput(), 0.001);
    }

    @Test
    public void testGetOutput () {
        assertEquals(0, neuron.getOutput(), 0.001);
    }

    @Test
    public void testSetOutput () {
        neuron.setOutput(0.1);
        assertEquals(0.1, neuron.getOutput(), 0.001);
    }

    @Test
    public void testGetDelta () {
        assertEquals(0.0, neuron.getDelta(), 0.001);
    }

    @Test
    public void testSetDelta () {
        neuron.setDelta(0.123);
        assertEquals(0.123, neuron.getDelta(), 0.001);
    }

    @Test
    public void testResetValues () {
        neuron.setDelta(0.1);
        neuron.setInput(0.1);
        neuron.setOutput(0.1);

        neuron.resetValues();

        assertEquals(0.0, neuron.getDelta(), 0.001);
        assertEquals(0.0, neuron.getInput(), 0.001);
        assertEquals(0.0, neuron.getOutput(), 0.001);
    }

    @Test
    public void testResetWeights () {
        
        List<Synapse> synapses = new ArrayList<Synapse>();
        Synapse synapse;

        for (int i = 0; i < 10; i += 1) {
            synapse = new DefaultSynapse(neuron, neuron, 0.1);
            synapse.setWeight(0.123);
            synapses.add(synapse);
            neuron.addOutgoingSynapse(synapse);
            /* Just in case this test runs before testGetWeight / testSetWeight */
            assertEquals(0.123, synapse.getWeight(), 0.001);
        }

        neuron.resetWeights();

        for (int i = 0; i < 10; i += 1) {
            synapse = synapses.get(i);
            assertEquals(0.0, synapse.getWeight(), 0.001);
        }

    }



    @Test
    public void testGetNeuronType () {
        assertEquals(NeuronType.Normal, neuron.getNeuronType());
    }

    @Test
    public void testSetNeuronType () {
        neuron.setNeuronType(NeuronType.Bias);
        assertEquals(NeuronType.Bias, neuron.getNeuronType());
    }


    /**
     * Test adding / getting / removing incoming synapses
     */

    @Test
    public void testGetIncomingSynapse () {
        Synapse synapse = neuron.getIncomingSynapse(0);
        assertNull(synapse);
    }

    @Test
    public void testAddIncomingSynapse () {
        Synapse synapse = new DefaultSynapse(neuron, neuron, 0.1);
        neuron.addIncomingSynapse(synapse);
        assertEquals(synapse, neuron.getIncomingSynapse(0));
    }

    @Test
    public void testRemoveIncomingSynapse () {
        neuron.removeIncomingSynapse(neuron.getIncomingSynapse(0));
        assertNull(neuron.getIncomingSynapse(0));
    }


    /**
     * Test adding / getting / removing outgoing synapses
     */

    @Test
    public void testGetOutgoingSynapse () {
        Synapse synapse = neuron.getOutgoingSynapse(0);
        assertNull(synapse);
    }

    @Test
    public void testAddOutgoingSynapse () {
        Synapse synapse = new DefaultSynapse(neuron, neuron, 0.1);
        neuron.addOutgoingSynapse(synapse);
        assertEquals(synapse, neuron.getOutgoingSynapse(0));
    }

    @Test
    public void testRemoveOutgoingSynapse () {
        neuron.removeOutgoingSynapse(neuron.getOutgoingSynapse(0));
        assertNull(neuron.getOutgoingSynapse(0));
    }


    @Test
    public void testSetActivationFunction () {
        neuron.setInput(1.0);
        neuron.setActivationFunction(new ActivationFunctionSigmoid());
        double output = neuron.calculateOutput();
        assertEquals(0.7310585786300049, output, 0.000000000001);
    }

    @Test
    public void testCalculateOutput () {
        double output = neuron.calculateOutput();
        assertEquals(0.0, output, 0.001);
    }

    @Test
    public void testGetOutgoingSynapseIterator () {
        neuron.addOutgoingSynapse(new DefaultSynapse(neuron, neuron, 1.0));
        Iterator<Synapse> outgoingSynapseIterator = neuron.getOutgoingSynapseIterator();
        Synapse synapse;
        while (outgoingSynapseIterator.hasNext()) {
            synapse = outgoingSynapseIterator.next();
            synapse.setValue(1.23);
            assertEquals(1.23, synapse.getValue(), 0.001);
        }
    }

}
