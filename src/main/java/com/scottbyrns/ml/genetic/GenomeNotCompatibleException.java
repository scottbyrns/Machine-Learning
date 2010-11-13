package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 10:24:13 AM
 */
public class GenomeNotCompatibleException extends Exception {

    public static final String NOT_SAME_LENGTH = "DefaultGenome length comparison error.";

    public GenomeNotCompatibleException (String message) {
        super(message);
    }
}
