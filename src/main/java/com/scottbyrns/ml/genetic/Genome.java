package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 11:28:05 AM
 */
public interface Genome {
    /**
     * Get the boolean[] representation of the genome.
     * @return
     */
    public boolean[] getGenome();

    /**
     * Set the boolean[] representation fo the genome.
     * @param genome
     * @return
     */
    public boolean[] setGenome(boolean[] genome);

    /**
     * Get the length of the genome.
     * @return
     */
    public int getLength ();
    /**
     * Return value of a gene at a specified index.
     * This will return a random boolean value if the index is out of bounds.
     * @param index
     * @return
     */
    public boolean getValueAt (int index);

    /**
     * Get the GenomeModel of the genome.
     * @return
     */
    public GenomeModel getModel ();
}
