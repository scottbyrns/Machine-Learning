package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 11:51:38 AM
 */
public interface Candidate {
    /**
     * Calculate the fitness of the candidates expression relative to a target.
     */
    public void calculateFitness();

    /**
     * Get the fitness of the candidate.
     * @return
     */
    public double getFitness();

    /**
     * Set the fitness of the candidate.
     * @param fitness
     * @return
     */
    public double setFitness(double fitness);

    /**
     * Get the CandidateModel of the candidate.
     * @return the candidate model being used by this candidate.
     */
    public CandidateModel getModel ();

    /**
     * Get the Genome of the candidate.
     * @return the candidates genome.
     */
    public Genome getGenome();
}
