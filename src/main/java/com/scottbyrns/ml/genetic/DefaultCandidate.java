package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 9:30:04 AM
 */
public abstract class DefaultCandidate implements Candidate {

    

    private Genome genome;

    private CandidateModel model;

    private double fitness;

    public DefaultCandidate(CandidateModel model) {
        setModel(model);
        genome = new DefaultGenome(model);
    }

    public DefaultCandidate(Candidate leftParent, Candidate rightParent) {
        setModel(leftParent.getModel());
        try {
            genome = new DefaultGenome(leftParent.getGenome(), rightParent.getGenome());
        }
        catch (GenomeNotCompatibleException exception) {
            /**
             *  @TODO do something meaningful with this exception
             */
            System.out.println("Watch what you breed. They are not compatible!");
        }
    }

    /**
     * Get the candidate fitness
     * @return
     */
    public double getFitness() {
        return fitness;
    }

    /**
     * Set the candidate fitness
     * @param fitness
     */
    public double setFitness(double fitness) {
        if (validateFitness(fitness)) {
            this.fitness = fitness;
        }
        return this.fitness;
    }

    /**
     * Check if the fitness is less than or equal to 1
     * @param fitness
     * @return
     */
    private static boolean validateFitness (double fitness) {
        if (fitness <= 1) {
            return true;
        }
        else {
            return false;
        }
    }

    /**
     * Get the CandidateModel used by the candidate.
     * @return
     */
    public CandidateModel getModel() {
        return this.model;
    }

    /**
     * Configure the candidate with an CandidateModel
     * @param model
     */
    private CandidateModel setModel(CandidateModel model) {
        this.model = model;
        return getModel();
    }

    /**
     * Get the candidates genome.
     * @return
     */
    public Genome getGenome() {
        return genome;
    }

    public void calculateFitness () {
        setFitness(fitnessCalculation());
    }

    /**
     * Calculate the fitness of the candidate with respect to it's implementation.
     * This should set DefaultCandidate.fitness to a float between 0.0f and 1.0f
     */
    public abstract double fitnessCalculation();

}
