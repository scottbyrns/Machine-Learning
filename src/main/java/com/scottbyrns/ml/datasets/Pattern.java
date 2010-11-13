package com.scottbyrns.ml.datasets;

import java.util.Vector;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 2:19:46 PM
 */
public interface Pattern {

    /**
     * Get the token used to split the input data set string.
     * @return String being used as the split token.
     */
    public String getDeliniater();

    /**
     * Set the token used to split the input data set string.
     * @param deliniater String to be used as the split token.
     */
    public void setDeliniater(String deliniater);

    /**
     * Set the input vector.
     * @param input
     */
    public void setInput (Vector<Double> input);

    /**
     * Get the input vector.
     * @return
     */
    public Vector<Double> getInput ();

    /**
     * Get the output vector.
     * @return
     */
    public Vector<Double> getOutput ();

}
