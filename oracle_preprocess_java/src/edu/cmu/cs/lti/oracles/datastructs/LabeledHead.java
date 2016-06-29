package edu.cmu.cs.lti.oracles.datastructs;

public class LabeledHead {

    public int headId;
    public String label;

    public LabeledHead(int headId, String label) {
        this.headId = headId;
        this.label = label;
    }

    public LabeledHead() {
        headId = -1; // default head of ROOT
        label = null; // default label TODO: make optional instead of null. Learn how to use Guava
                      // for this.
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + headId;
        result = prime * result + ((label == null) ? 0 : label.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        LabeledHead other = (LabeledHead) obj;
        if (headId != other.headId)
            return false;
        if (label == null) {
            if (other.label != null)
                return false;
        } else if (!label.equals(other.label))
            return false;
        return true;
    }

    @Override
    public String toString() {
        return "LabeledHead [headId=" + headId + ", label=" + label + "]";
    }

}
