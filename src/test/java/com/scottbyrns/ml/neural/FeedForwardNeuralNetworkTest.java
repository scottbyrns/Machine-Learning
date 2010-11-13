package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.datasets.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Vector;

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 2:02:16 PM
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
    public void testMeasurePatternListError () {
        /**
         * @TODO see if this can be tested better. The outcome of this method changes each time
         * so I am only testing to see if an error has occured.
         */
        Vector<Double> output = new Vector<Double>();
        output.add(1.3);
        output.add(0.2);

        Vector<Pattern> patternList = new Vector<Pattern>();
        patternList.add(new DefaultPattern(output, output));

        double error = feedForwardNetwork.measurePatternListError(patternList);
        assertTrue(error != (-1.0));
    }

    @Test
    public void testMeansSquaredError () {
        Vector<Double> output = new Vector<Double>();
        output.add(1.3);
        output.add(0.2);
        double error = feedForwardNetwork.meanSquaredError(output);
        assertEquals(0.8650000000000001, error, 0.0000000001);
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
