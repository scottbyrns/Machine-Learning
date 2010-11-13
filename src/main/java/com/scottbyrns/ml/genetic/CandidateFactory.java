package com.scottbyrns.ml.genetic;

import java.io.Serializable;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 11:03:08 AM
 */
public interface CandidateFactory extends Serializable {
    /**
     * Return a new instance of a DefaultCandidate.
     * @param model
     * @return
     */
    public Candidate createCandidate (CandidateModel model);

    /**
     * Breed two candidates together to generate a new candidate based on the parent candidate genomes.
     * @param leftParent
     * @param rightParent
     * @return
     */
	public Candidate breedCandidates (Candidate leftParent, Candidate rightParent);
}

