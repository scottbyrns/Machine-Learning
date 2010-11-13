package com.scottbyrns.ml.genetic;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 9:30:20 AM
 */
public class CandidateTest {

    private Candidate candidate;

    private CandidateModel mockCandidateModel = new MockCandidateModel();

    @Before
    public void setup () {
        candidate = new DefaultCandidate(mockCandidateModel) {
            public double fitnessCalculation () {
                return 0.1;
            }
        };
    }

    @Test
    public void testGetFitness () {
        assertEquals(0, candidate.getFitness(), 0.001);
    }

    @Test
    public void testSetFitness () {
        assertEquals(0.99, candidate.setFitness(0.99), 0.001);
    }

    @Test
    public void testSetFitnessGreaterThanOne () {
        assertFalse(1.1 == candidate.setFitness(1.1));
    }

    @After
    public void teardown () {
        candidate = null;
    }

}
