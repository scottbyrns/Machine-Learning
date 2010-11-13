package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 10:58:08 AM
 */
public interface PopulationModel extends CandidateModel {
	/**
	 * @return Number of genetic candidates to create.
	 */
	public int getPopulationSize ();
	/**
	 * @return Number of generations the population will exist for before execution ends.
	 */
	public int getGenerations ();
    /**
     * @return Percentage of population to retain per generation.
     */
    public float getPopulationCutoff ();
}
