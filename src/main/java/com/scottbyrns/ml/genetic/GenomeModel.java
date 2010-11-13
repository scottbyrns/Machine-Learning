package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 10:37:17 AM
 */
public interface GenomeModel {
	/**
     * @return Length of the genome required to express a meaningful representation.
	 */
	public int getGenomeLength ();
	/**
	 * @return Length of a "gene" in the genome to be cut out and added to a child genome.
	 */
	public int getGenomeCrossLength ();
	/**
	 * @return Require a uniform cross, or that genes copied are equally distributed between the parents.
	 * If this is false a parent is chosen at random each time.
	 */
	public boolean isUniformCross ();
	/**
	 * @return Chance of mutating when cross breeding parents.
	 */
	public float getMutateChance ();
	/**
	 * @return How much mutation may occur.
	 */
	public float getMutateAmount ();
}
