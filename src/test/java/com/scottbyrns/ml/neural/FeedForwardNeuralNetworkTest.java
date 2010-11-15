package com.scottbyrns.ml.neural;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * @author Scott Byrns
 * Date: Nov 11, 2010
 * Time: 2:02:16 PM
 *
 * @version 1.0
 */
public class FeedForwardNeuralNetworkTest {

    FeedForwardNeuralNetwork feedForwardNetwork;

    @Before
    public void setup () {
        feedForwardNetwork = new DefaultFeedForwardNeuralNetwork(3, new int[]{1}, 2);
        feedForwardNetwork = new DefaultFeedForwardNeuralNetwork(feedForwardNetwork);
    }

    @Test
    public void testConnectNeurons () {



        /**
         * @TODO implement
         */
    }

    @Test
    public void testGetNumberNeuronsInput () {
        assertEquals(3, feedForwardNetwork.getNumberNeuronsInput(NeuronType.Normal));
    }

    @Test
    public void testGetNumberNeuronsHidden () {
        assertEquals(1, feedForwardNetwork.getNumberNeuronsHidden(NeuronType.Normal).length);
    }

    @Test
    public void testGetNumberNeuronsOutput () {
        assertEquals(2, feedForwardNetwork.getNumberNeuronsOutput(NeuronType.Normal));
    }

    @Test
    public void testGetOutputNeurons () {
        /**
         * @TODO Write this test.
         */
    }

    @After
    public void teardown () {
        feedForwardNetwork = null;
    }


}
