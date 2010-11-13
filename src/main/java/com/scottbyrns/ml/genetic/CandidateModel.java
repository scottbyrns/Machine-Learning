package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 9:57:01 AM
 */
public interface CandidateModel extends GenomeModel {
    /**
     * @return Target fitness cutoff. If the fitness of a candidate is equal to or greater than this value
     * an OptimizationCutoffReached exception will be thrown.
     */
    public double getFitnessCutoff ();
}