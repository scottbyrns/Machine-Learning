package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunctionLinear;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 6:07:08 PM
 */
public class SynapseLayerTest {

    private SynapseLayer synapseLayer;
    private Neuron neuron;
    private Synapse synapse;

    @Before
    public void setup () {
        neuron = new DefaultNeuron(new ActivationFunctionLinear());
        synapse = new DefaultSynapse(neuron, neuron, 0.123);
        synapseLayer = new DefaultSynapseLayer();
    }

    @Test
    public void testGetSynapseAtIndex () {
        assertNull(synapseLayer.getSynapseAtIndex(0));
    }

    @Test
    public void testAdd () {
        synapseLayer.add(synapse);
        assertEquals(synapse, synapseLayer.getSynapseAtIndex(0));
    }


    @After
    public void teardown () {

    }    

}
