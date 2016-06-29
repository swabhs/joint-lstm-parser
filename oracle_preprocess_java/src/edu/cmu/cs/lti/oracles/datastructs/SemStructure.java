package edu.cmu.cs.lti.oracles.datastructs;

import java.util.Map;

import com.google.common.collect.Maps;

public class SemStructure {

    private int predPos;
    private String pred;
    private Map<Integer, SemHead> args;

    public SemStructure(int predPos, String pred, Map<Integer, SemHead> args) {
        this.predPos = predPos;
        this.pred = pred;
        this.args = args;
    }

    public SemStructure(int predPos, String pred) {
        this.predPos = predPos;
        this.pred = pred;
        this.args = Maps.newTreeMap();
    }

    public void addArg(int pos, String argType) {
        args.put(pos, new SemHead(predPos, argType));
    }

    public int getPredPos() {
        return predPos;
    }

    public String getPred() {
        return pred;
    }

    public SemHead getSemHead(int arg) {
        if (args.containsKey(arg)) {
            return args.get(arg);
        }
        return null; // TODO: make more elegant
    }

    public boolean isArg(int argPos) {
        return args.containsKey(argPos);
    }

    public Map<Integer, SemHead> getAllArgs() {
        return args;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((args == null) ? 0 : args.hashCode());
        result = prime * result + ((pred == null) ? 0 : pred.hashCode());
        result = prime * result + predPos;
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        SemStructure other = (SemStructure) obj;

        if (pred == null) {
            if (other.pred != null)
                return false;
        } else if (!pred.equals(other.pred))
            return false;
        if (predPos != other.predPos)
            return false;

        for (Map.Entry<Integer, SemHead> entry : args.entrySet()) {
            // System.err.println("argument at " + entry.getKey());
            if (other.args.containsKey(entry.getKey()) == false) {
                // System.err.println("argument not found");
                return false;
            }
            if (other.args.get(entry.getKey()).equals(entry.getValue()) == false) {
                // System.err.println("mislabeled arguemnt");
                return false;
            }
        }
        return true;
    }

}
