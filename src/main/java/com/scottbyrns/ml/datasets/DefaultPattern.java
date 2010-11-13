package com.scottbyrns.ml.datasets;

import java.util.StringTokenizer;
import java.util.Vector;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 2:19:31 PM
 */
public class DefaultPattern implements Pattern {

    private String deliniater = ";";

    private Vector<Double> input;
    private Vector<Double> output;

    private void populateInputVectorFromDoubleArray (double[] input) {
        setInput(new Vector<Double>());
        for (int i = 0; i < input.length; i += 1) {
            getInputVector().add(input[i]);
        }
    }

    private void populateOutputVectorFromDoubleArray (double[] output) {
        setOutputVector(new Vector<Double>());
        for (int i = 0; i < output.length; i += 1) {
            getOutputVector().add(output[i]);
        }
    }

    /**
     * DefaultPattern constructor for unsupervised patterns
     * @param first input vector.
     */
    public DefaultPattern(double[] first) {
        populateInputVectorFromDoubleArray(first);
    }

    /**
     * DefaultPattern constructor for unsupervised patterns
     * @param first input vector.
     */
    public DefaultPattern(Vector<Double> first) {
        setInputVector(first);
    }

    /**
     * DefaultPattern constructor for unsupervised patterns
     * @param first input vector.
     */
    public DefaultPattern(String first) {
        setInputVector(parsePatternString(first));
    }

    /**
     * DefaultPattern constructor for supervised patterns
     * @param first Input vector.
     * @param second Output vector.
     */
    public DefaultPattern(double[] first, double[] second) {
        populateInputVectorFromDoubleArray(first);
        populateOutputVectorFromDoubleArray(second);
    }

    /**
     * DefaultPattern constructor for supervised patterns
     * @param first input vector.
     * @param second Output vector.
     */
    public DefaultPattern(Vector<Double> first, Vector<Double> second) {
        setInputVector(first);
        setOutputVector(second);
    }

    /**
     * DefaultPattern constructor for supervised patterns
     * @param first input vector.
     * @param second Output vector.
     */
    public DefaultPattern(String first, String second) {
        setInputVector(parsePatternString(first));
        setOutputVector(parsePatternString(second));
    }

    /**
     * DefaultPattern constructor for supervised patterns
     * @param first input vector.
     * @param second Output vector.
     * @param token
     */
    public DefaultPattern(String first, String second, String token) {
        setDeliniater(token);
        setInputVector(parsePatternString(first));
        setOutputVector(parsePatternString(second));
    }



    private Vector<Double> parsePatternString (String string) {
        Vector<Double> list = new Vector<Double>();
        StringTokenizer tokenizer = new StringTokenizer(string, getDeliniater());
        
        while (tokenizer.hasMoreTokens()) {
            list.add(Double.parseDouble(tokenizer.nextToken()));
        }

        return list;
    }




    public void setInput(Vector<Double> input) {
        setInputVector(input);
    }

    public Vector<Double> getInput () {
        return getInputVector();
    }

    public Vector<Double> getOutput() {
        return getOutputVector();
    }




    private void setInputVector (Vector<Double> input) {
        this.input = input;
    }

    private Vector<Double> getInputVector () {
        return this.input;
    }

    private void setOutputVector (Vector<Double> output) {
        this.output = output;
    }

    private Vector<Double> getOutputVector () {
        return this.output;
    }


    /**
     * Get the token used to split the input data set string.
     * @return
     */
    public String getDeliniater() {
        return deliniater;
    }

    /**
     * Set the token used to split the input data set string.
     * @param deliniater
     */
    public void setDeliniater(String deliniater) {
        this.deliniater = deliniater;
    }
}
