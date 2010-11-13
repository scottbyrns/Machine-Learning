package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.Mathematics;
import com.scottbyrns.ml.neural.Activation.ActivationFunctionConstant;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 9:20:10 AM
 */
public class SynapseTest {

    private Synapse synapse;

    Neuron testNeuronSource;
    Neuron testNeuronDestination;

    @Before
    public void setup () {

        testNeuronDestination = new DefaultNeuron(new ActivationFunctionConstant());
        testNeuronSource = new DefaultNeuron(new ActivationFunctionConstant());

        synapse = new DefaultSynapse(testNeuronSource, testNeuronDestination, Mathematics.rand());
        
    }

    /**
     * Test Input neuron getter / setter
     */

    @Test
    public void testGetInputNeuron () {
        assertEquals(testNeuronSource, synapse.getInputNeuron());
    }

    @Test
    public void testSetInputNeuron () {
        Neuron testNeuron = new DefaultNeuron(new ActivationFunctionConstant());
        synapse.setInputNeuron(testNeuron);

        assertEquals(testNeuron, synapse.getInputNeuron());
    }

    /**
     * Test Output neuron getter / setter
     */

    @Test
    public void testGetOutputNeuron () {
        assertEquals(testNeuronDestination, synapse.getOutputNeuron());
    }

    @Test
    public void testSetOutputNeuron () {
        Neuron testNeuron = new DefaultNeuron(new ActivationFunctionConstant());
        synapse.setOutputNeuron(testNeuron);

        assertEquals(testNeuron, synapse.getOutputNeuron());
    }

    /**
     * Test Weight getter / setter
     */

    @Test
    public void testGetWeight () {
        synapse.setWeight(0.0);
        assertEquals(0.0, synapse.getWeight(), 0.001);
    }

    @Test
    public void testSetWeight () {
        synapse.setWeight(0.123);
        assertEquals(0.123, synapse.getWeight(), 0.001);
    }

    @Test
    public void testResetWeight () {
        synapse.resetWeight();
        assertEquals(0.0, synapse.getWeight(), 0.001);
    }

    /**
     * Test Value getter / setter
     */

    @Test
    public void testGetValue () {
        assertEquals(0.0, synapse.getValue(), 0.001);
    }

    @Test
    public void testSetValue () {
        synapse.setValue(0.123);
        assertEquals(0.123, synapse.getValue(), 0.001);
    }
}
