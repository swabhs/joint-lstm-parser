package edu.cmu.cs.lti.oracles.datastructs;

import java.util.List;

public class ConllElement {

    public int id;
    public String form;
    public String predictedLemma;
    public String predictedPOS;
    // TODO: add code for feat?
    public int head;
    public String depRel;
    public int predictedHead;
    public String predictedDepRel;
    public boolean isPred;
    public String pred;
    public List<String> apreds;

}
