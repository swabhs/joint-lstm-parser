package edu.cmu.cs.lti.oracles.datastructs;

public class SemHead implements Comparable<SemHead> {

    public int semHeadId;
    public String semLabel;

    public SemHead(int headId, String label) {
        this.semHeadId = headId;
        this.semLabel = label;
    }

    public SemHead() {
        semHeadId = -1; // default head of ROOT
        semLabel = null; // default label TODO: make optional instead of null. Learn how to use
                         // Guava for this.
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + semHeadId;
        result = prime * result + ((semLabel == null) ? 0 : semLabel.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        SemHead other = (SemHead) obj;
        if (semHeadId != other.semHeadId)
            return false;
        if (semLabel == null) {
            if (other.semLabel != null)
                return false;
        } else if (!semLabel.equals(other.semLabel))
            return false;
        return true;
    }

    @Override
    public String toString() {
        return semHeadId + "-" + semLabel + "->";
    }

    @Override
    public int compareTo(SemHead o) {
        return Integer.compare(this.semHeadId, o.semHeadId);
    }

}
