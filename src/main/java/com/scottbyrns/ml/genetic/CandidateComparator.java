package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 9:59:50 AM
 */
import java.util.Comparator;

public class CandidateComparator implements Comparator<Candidate> {
	public int compare (Candidate leftCandidate, Candidate rightCandidate) {
		if (leftCandidate.getFitness() > rightCandidate.getFitness()) {
			return -1;
		}
		else if (leftCandidate.getFitness() < rightCandidate.getFitness()) {
			return 1;
		}
		else {
			return 0;
		}
	}
}