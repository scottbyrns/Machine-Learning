package com.scottbyrns.ml.datasets;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 10:43:41 PM
 */
public class PatternSetTest {

    private PatternSet patternSet;

    @Before
    public void setup () {
        patternSet = new DefaultPatternSet();
    }

    @Test
    public void testAddPatternAndTheGetSetGetters () {
        int count = 100;
        while (count-- > 0) {
            patternSet.addPattern(new DefaultPattern(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
        }

        List<Pattern> a = patternSet.getTrainingSet();
        
        assertEquals(60, patternSet.getTrainingSet().size());
        assertEquals(30, patternSet.getValidationSet().size());
        assertEquals(10, patternSet.getTestSet().size());

    }

    @After
    public void teardown () {
        patternSet = null;
    }

}
