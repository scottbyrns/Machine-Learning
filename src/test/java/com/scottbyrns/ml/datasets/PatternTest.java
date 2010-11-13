package com.scottbyrns.ml.datasets;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Vector;

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 7:40:06 PM
 */
public class PatternTest {

    private Pattern pattern;

    @Before
    public void setup () {
        pattern = new DefaultPattern(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    }

    @Test
    public void testGetDeliniater () {
        assertEquals(";", pattern.getDeliniater());
    }

    @Test
    public void testSetDeliniater () {
        pattern.setDeliniater(":");
        assertEquals(":", pattern.getDeliniater());
    }

    @Test
    public void testGetInput () {
        Vector<Double> inputVector = pattern.getInput();
        double[] input = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

        for (int i = 0; i < input.length; i += 1) {
            assertEquals(input[i], inputVector.get(i), 0.001);
        }

    }

    @Test
    public void testSetInput () {
        Vector<Double> inputVector = new Vector<Double>();
        double[] input = new double[]{6.0, 5.0, 4.0, 3.0, 2.0, 1.0};

        for (double inputValue : input) {
            inputVector.add(inputValue);
        }

        pattern.setInput(inputVector);

        for (int i = 0; i < input.length; i += 1) {
            assertEquals(input[i], pattern.getInput().get(i), 0.001);
        }
    }

    @Test
    public void testGetOutputIsNull () {
        assertNull(pattern.getOutput());
    }

    @Test
    public void testGetOutput () {
        pattern = new DefaultPattern(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, new double[]{1.0, 2.0, 3.0});

        Vector<Double> outputVector = pattern.getOutput();

        double[] output = new double[]{1.0, 2.0, 3.0};

        for (int i = 0; i < output.length; i += 1) {
            assertEquals(output[i], outputVector.get(i), 0.001);
        }
    }


    @After
    public void teardown () {
        
    }

}
