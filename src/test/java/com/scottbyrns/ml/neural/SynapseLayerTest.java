package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunctionLinear;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Iterator;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

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
    public void testGetSynapseIterator () {
        Iterator<Synapse> synapseIterator = synapseLayer.getSynapsesIterator();
        assertNotNull(synapseIterator);
    }

    @Test
    public void testAdd () {
        synapseLayer.addSynapse(synapse);
        Iterator<Synapse> synapseIterator = synapseLayer.getSynapsesIterator();
        assertTrue(synapseIterator.hasNext());
    }


    @After
    public void teardown () {
        neuron = null;
        synapse = null;
        synapseLayer = null;
    }    

}
