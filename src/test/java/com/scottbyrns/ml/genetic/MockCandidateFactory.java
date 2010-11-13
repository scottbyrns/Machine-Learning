package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 12:08:37 PM
 */
public class MockCandidateFactory implements CandidateFactory {
    public Candidate createCandidate (CandidateModel model) {
        return new MockCandidate(model);
    }
    public Candidate breedCandidates (Candidate leftParent, Candidate rightParent) {
        return new MockCandidate(leftParent, rightParent);
    }
}
