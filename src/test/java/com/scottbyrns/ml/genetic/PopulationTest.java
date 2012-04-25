package com.scottbyrns.ml.genetic;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 11:54:54 AM
 */
public class PopulationTest {

    private PopulationModel mockCandidateModel = new MockCandidateModel();

    CandidateFactory mockCandidateFactory = new MockCandidateFactory();

    Population population;

    @Before
    public void setup () {
        population = new DefaultPopulation(mockCandidateModel, mockCandidateFactory);
    }

    @Test
    public void testNextGeneration () {
        try {
            population.nextGeneration();
            assertTrue(true);
        }
        catch (OptimizationCutoffException e) {
            assertTrue(true);
        }
        catch (Exception e) {
            e.printStackTrace();
            assertTrue(false);
        }
    }

    @Test
    public void testGetFittest () {
        Candidate candidate = population.getFittest();
        assertNotNull(candidate);
    }

    @Test
    public void testIsEvolving () {
        Candidate candidate = population.getFittest();
        double fitness = candidate.getFitness();
        while(population.getFittest().getFitness() == fitness) {
            try {
                population.nextGeneration();
            }
            catch (OptimizationCutoffException e) {
                assertTrue(true);
                return;
            }
        }
        candidate = population.getFittest();
        boolean is = (fitness < candidate.getFitness());
        assertTrue(is);
    }


    @Test
    public void testWillOptimize () {
        System.out.println("Hold your horses this test is going to take a bit to run.");
        Candidate candidate = population.getFittest();
        while(true) {
            try {
                population.nextGeneration();
            }
            catch (OptimizationCutoffException e) {
                assertTrue(true);
                return;
            }
        }
    }

}
