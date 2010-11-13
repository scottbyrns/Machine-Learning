package com.scottbyrns.ml.genetic;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 9:31:52 AM
 */
public class GenomeTest {

    private CandidateModel mockCandidateModel = new MockCandidateModel();

    private Genome genome;

    @Before
    public void setup () {
        genome = new DefaultGenome(mockCandidateModel);
    }

    @Test
    public void testGetGenome () {
        assertEquals(mockCandidateModel.getGenomeLength(), genome.getGenome().length);
    }

    @Test
    public void testSetGenome () {
        Genome genome = new DefaultGenome(mockCandidateModel);
        assertEquals(genome.getGenome(), this.genome.setGenome(genome.getGenome()));
    }
    
    @After
    public void teardown () {
        genome = null;
    }
}
