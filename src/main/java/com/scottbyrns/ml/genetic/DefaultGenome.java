package com.scottbyrns.ml.genetic;

import java.util.Random;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 9:31:02 AM
 */
public class DefaultGenome implements Genome {

    private boolean[] genome;
    private Random random = new Random();

    private int length = 0;
    private float mutateChance;
    private float mutateAmount;
    private boolean uniformCross;
    private int genomeCrossLength;

    private GenomeModel model;

    public DefaultGenome(GenomeModel model) {
        setModel(model);
        createGenome();
    }

    public DefaultGenome(Genome leftGenome, Genome rightGenome) throws GenomeNotCompatibleException {
        setModel(leftGenome.getModel());
        setGenome(breedGenome(leftGenome, rightGenome));
    }

    public boolean[] getGenome() {
        return genome;
    }



    public boolean[] setGenome(boolean[] genome) {
        this.genome = genome;
        return getGenome();
    }

    public GenomeModel getModel() {
        return model;
    }

    public void setModel(GenomeModel model) {
        this.model = model;
        setLength(model.getGenomeLength());
        setMutateAmount(model.getMutateAmount());
        setMutateChance(model.getMutateChance());
        setUniformCross(model.isUniformCross());
        setGenomeCrossLength(model.getGenomeCrossLength());
    }

    public int getLength () {
        return length;
    }

    private int setLength (int length) {
        this.length = length;
        return this.length;
    }

    private int getGenomeCrossLength() {
        return genomeCrossLength;
    }

    private void setGenomeCrossLength(int genomeCrossLength) {
        this.genomeCrossLength = genomeCrossLength;
    }

    private float getMutateChance() {
        return mutateChance;
    }

    private void setMutateChance(float mutateChance) {
        this.mutateChance = mutateChance;
    }

    private float getMutateAmount() {
        return mutateAmount;
    }

    private void setMutateAmount(float mutateAmount) {
        this.mutateAmount = mutateAmount;
    }

    private boolean isUniformCross() {
        return uniformCross;
    }

    private void setUniformCross(boolean uniformCross) {
        this.uniformCross = uniformCross;
    }

    /**
     * Create the genome.
     * @return True if created, false if already created.
     */
    private boolean createGenome () {
        if (null == genome) {
            genome = new boolean[this.length];

            for (int i = 0; i < this.length; i += 1) {
                this.genome[i] = randomBool();
            }

            return true;
        }
        else {
            return false;
        }
    }

    /**
     * Return value of a gene at a specified index.
     * This will return a random boolean value if the index is out of bounds.
     * @param index
     * @return
     */
    public boolean getValueAt (int index) {
        if (index > getLength() - 1) {
            return randomBool();
        }
        return this.genome[index];
    }

    /**
     * Take two genomes and splice them together to create a unique genome.
     * @param leftGenome
     * @param rightGenome
     * @return
     * @throws GenomeNotCompatibleException
     */
    private boolean[] breedGenome (Genome leftGenome, Genome rightGenome) throws GenomeNotCompatibleException {
        if (leftGenome.getLength() != rightGenome.getLength()) {
            throw new GenomeNotCompatibleException(GenomeNotCompatibleException.NOT_SAME_LENGTH);
        }

        boolean[] output = new boolean[getLength()];

        int cross = random.nextInt(getGenomeCrossLength());
        Genome parent;

        for (int i = 0; i < getLength(); i += genomeCrossLength) {

            if (isUniformCross()) {
                if (randomBool()) {
                    parent = leftGenome;
                }
                else {
                    parent = rightGenome;
                }
            }
            else {
                if (i < cross) {
                    parent = leftGenome;
                }
                else {
                    parent = rightGenome;
                }
            }


            for (int j = 0; j < genomeCrossLength; j += 1) {
                output[i+j] = parent.getValueAt(i+j);

                if (random.nextFloat() < getMutateChance()) {
                    double mutation = Math.random() * 10 * mutateAmount;
                    if (mutation < 0.5) {
                        output[i+j] = false;
                    }
                    else {
                        output[i+j] = true;
                    }
                }
            }


        }
        return output;
    }

    private boolean randomBool () {
        return random.nextBoolean();
    }

}
