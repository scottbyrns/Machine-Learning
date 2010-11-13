package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 11:35:51 AM
 */
public interface Population {
    /**
	 * Create the next generation of the population.
	 */
    public void nextGeneration () throws OptimizationCutoffReached;
    /**
     * Sort the candidates by fitness and return the fittest of them.
     * @return The fittest candidate
     */
    public Candidate getFittest ();
    /**
     * Randomly sort our population.
     */
    public void randomSortCandidates ();
}
