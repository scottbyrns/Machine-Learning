package com.scottbyrns.ml.genetic;

import java.util.Comparator;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 11:19:03 AM
 */
public class RandomSortCandidateComparator implements Comparator<Candidate> {
	public int compare (Candidate leftCandidate, Candidate rightCandidate) {
        int check = (int)(Math.random() * 3);
        return check - 1;
	}
}
