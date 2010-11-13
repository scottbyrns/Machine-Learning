package com.scottbyrns.ml.genetic;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 9:27:32 AM
 */
public class DefaultPopulation implements Population {

    private PopulationModel model;
    private boolean candidatesSorted;

    private Candidate[] candidates;
    private CandidateFactory candidateFactory;

    public DefaultPopulation(PopulationModel model, CandidateFactory candidateFactory) {
        setModel(model);
        setCandidateFactory(candidateFactory);
        createPopulation();
    }

    /**
	 * Create the population of candidates.
	 */
    private void createPopulation () {
        candidates = new DefaultCandidate[getModel().getPopulationSize()];
        for (int i = 0; i < getModel().getPopulationSize(); i += 1) {
            candidates[i] = getCandidateFactory().createCandidate(getModel());
        }
    }

	/**
	 * Sort the population by its fitness.
	 */
	private void sortPopulationByFitness () {
        if (isCandidatesSorted()) {
            return;
        }

        for (int i = 0; i < getModel().getPopulationSize(); i += 1) {
            candidates[i].calculateFitness();
        }

		Arrays.sort(candidates, new CandidateComparator());
        setCandidatesSorted(true);
	}

	/**
	 * Create the next generation of the population.
	 */
	public void nextGeneration () throws OptimizationCutoffReached {

		Random random = new Random();

		sortPopulationByFitness();

		int populationToReserve = (int)(getModel().getPopulationCutoff() * getModel().getPopulationSize());

		for (int i = populationToReserve; i < getModel().getPopulationSize(); i += 1) {
            candidates[i] = null;
			candidates[i] = getCandidateFactory().breedCandidates(
					candidates[(int)(random.nextFloat() * populationToReserve)],
					candidates[(int)(random.nextFloat() * populationToReserve)]
				);
		}

        setCandidatesSorted(false);

        if (getModel().getPopulationCutoff() <= getFittest().getFitness()) {
            throw new OptimizationCutoffReached("Optimization cutoff reached");
        }

    }

    /**
     * Sort the candidates by fitness and return the fittest of them.
     * @return The fittest candidate
     */
    public Candidate getFittest () {
        sortPopulationByFitness();
        return candidates[0];
    }

    /**
     * Randomly sort our population.
     */
    public void randomSortCandidates () {
        Arrays.sort(candidates, new RandomSortCandidateComparator());
    }






    
    private PopulationModel getModel() {
        return model;
    }

    private void setModel(PopulationModel model) {
        this.model = model;
    }

    private Candidate[] getCandidates() {
        return candidates;
    }

    private void setCandidates(DefaultCandidate[] defaultCandidates) {
        this.candidates = defaultCandidates;
    }

    private CandidateFactory getCandidateFactory() {
        return candidateFactory;
    }

    private void setCandidateFactory(CandidateFactory candidateFactory) {
        this.candidateFactory = candidateFactory;
    }

    private boolean isCandidatesSorted() {
        return candidatesSorted;
    }

    private void setCandidatesSorted(boolean candidatesSorted) {
        this.candidatesSorted = candidatesSorted;
    }

}
