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

    @Test
    public void testAddIncomingSynapse () {
        Synapse synapse = new DefaultSynapse(neuron, neuron, 0.1);
        neuron.addIncomingSynapse(synapse);

        Iterator<Synapse> incomingSynapseIterator = neuron.getIncomingSynapseIterator();
        assertEquals(synapse, incomingSynapseIterator.next());

    }

    @Test
    public void testGetIncomingSynapseIterator () {
        Iterator<Synapse> incomingSynapseIterator = neuron.getIncomingSynapseIterator();
        assertNotNull(incomingSynapseIterator);
    }

    @Test
    public void testRemoveIncomingSynapse () {

        Synapse synapse = new DefaultSynapse(neuron, neuron, 0.1);
        neuron.addIncomingSynapse(synapse);

        Iterator<Synapse> incomingSynapseIterator = neuron.getIncomingSynapseIterator();

        neuron.removeIncomingSynapse(incomingSynapseIterator.next());

        incomingSynapseIterator = neuron.getIncomingSynapseIterator();

        assertFalse(incomingSynapseIterator.hasNext());
    }

    @Test
    public void testAddOutgoingSynapse () {
        Synapse synapse = new DefaultSynapse(neuron, neuron, 0.1);
        neuron.addOutgoingSynapse(synapse);

        Iterator<Synapse> outgoingSynapseIterator = neuron.getOutgoingSynapseIterator();

        assertEquals(synapse, outgoingSynapseIterator.next());
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


    @Test
    public void testRemoveOutgoingSynapse () {
        Synapse synapse = new DefaultSynapse(neuron, neuron, 0.1);
        neuron.addOutgoingSynapse(synapse);

        Iterator<Synapse> outgoingSynapseIterator = neuron.getOutgoingSynapseIterator();

        neuron.removeOutgoingSynapse(outgoingSynapseIterator.next());

        outgoingSynapseIterator = neuron.getOutgoingSynapseIterator();
        assertFalse(outgoingSynapseIterator.hasNext());
    }

    @Test
    public void testSetActivationFunction () {
        neuron.setInput(1.0);
        neuron.setActivationFunction(new ActivationFunctionSigmoid());
        double output = neuron.calculateOutput();
        assertEquals(new ActivationFunctionSigmoid().calculate(1.0), output, 0.000000000001);
    }

    @Test
    public void testCalculateOutput () {
        neuron.setInput(0.123);
        neuron.setActivationFunction(new ActivationFunctionSigmoid());
        double output = neuron.calculateOutput();
        assertEquals(new ActivationFunctionSigmoid().calculate(0.123), output, 0.001);
    }

    @Test
    public void testCalculateDerivative () {
        neuron.setInput(0.123);
        neuron.setActivationFunction(new ActivationFunctionSigmoid());
        double output = neuron.calculateDerivative(neuron.getOutput());
        assertEquals(new ActivationFunctionSigmoid().calculateDerivate(neuron.getOutput()), output, 0.001);
    }

}
