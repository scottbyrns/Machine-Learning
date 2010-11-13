package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 12:05:47 PM
 */
public class MockCandidate extends DefaultCandidate {

	public MockCandidate (CandidateModel model) {
		super(model);
	}

	public MockCandidate (Candidate leftParent, Candidate rightParent) {
		super(leftParent, rightParent);
	}

	public double fitnessCalculation () {
        char[] helloworld = new char[] {'h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd'};
        int charsChecked = 0;
        int charsCorrect = 0;

        double modifyer = 0.0d;

        String generateString = "";

        for (int i = 0; i < getGenome().getLength()-6; i += 7) {
            String geneString = "";
            int j = 0;
            while ((j += 1) < 7) {
                geneString += (getGenome().getValueAt(i+j)) ? 1 : 0;
            }
            int geneInt = Integer.parseInt(geneString, 2) + 97;



            char generated = (char)geneInt;
            char provided = helloworld[charsChecked];

            generateString += Character.toString(generated);

            charsChecked += 1;
            if (generated == provided) {
                charsCorrect += 1;
            }

            /* Positively weight characters that are in the range of a-z */
            if (geneInt < 123 && geneInt > 96) {
                modifyer += 0.009d;
            }


        }

        return ((double)charsCorrect/(double)charsChecked) + modifyer;

    }
}
