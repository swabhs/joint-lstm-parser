package edu.cmu.cs.lti.oracles.datastructs;

import java.util.List;

public class Conll {

    public List<ConllElement> elements;
    public int size;

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (ConllElement ele : elements) {
            builder.append(ele.toString());
            // builder.append("\n");
        }
        return builder.toString();
    }

}
