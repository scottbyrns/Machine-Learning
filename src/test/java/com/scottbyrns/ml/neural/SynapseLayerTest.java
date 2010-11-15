package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Activation.ActivationFunctionLinear;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Iterator;
import java.util.List;
import java.util.Vector;

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
    public void testGetSynapseIterator () {
        Iterator<Synapse> synapseIterator = synapseLayer.getSynapsesIterator();
        assertNotNull(synapseIterator);
    }

    @Test
    public void testAddSynapse () {
        synapseLayer.addSynapse(synapse);
        Iterator<Synapse> synapseIterator = synapseLayer.getSynapsesIterator();
        assertTrue(synapseIterator.hasNext());
    }

    @Test
    public void testGetWeightVector () {
        synapseLayer.addSynapse(synapse);
        synapseLayer.addSynapse(synapse);

        List<Double> weightVector = synapseLayer.getWeightVector();

        Iterator<Double> weightVectorIterator = weightVector.iterator();
        while (weightVectorIterator.hasNext()) {
            assertEquals(0.123, weightVectorIterator.next(), 0.001);
        }
    }

    @Test
    public void testSetWeightVector () {
        synapseLayer.addSynapse(synapse);
        synapseLayer.addSynapse(synapse);

        Vector<Double> newWeightVector = new Vector<Double>();
        newWeightVector.add(0.321);
        newWeightVector.add(0.321);

        synapseLayer.setWeightVector(newWeightVector.iterator());

        List<Double> weightVector = synapseLayer.getWeightVector();

        Iterator<Double> weightVectorIterator = weightVector.iterator();
        while (weightVectorIterator.hasNext()) {
            assertEquals(0.321, weightVectorIterator.next(), 0.001);
        }
    }

    @After
    public void teardown () {
        neuron = null;
        synapse = null;
        synapseLayer = null;
    }    

}
