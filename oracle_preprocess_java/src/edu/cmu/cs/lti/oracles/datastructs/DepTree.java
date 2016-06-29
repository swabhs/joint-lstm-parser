package edu.cmu.cs.lti.oracles.datastructs;

import java.util.Collection;
import java.util.List;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;

public class DepTree {

    private List<LabeledHead> headsAndDeps;
    private List<LabeledHead> predictedHeadsAndDeps; // required for a pipelined model
    private Multimap<Integer, Integer> children;
    public int size;

    public DepTree(int size) {
        this.size = size;
        this.headsAndDeps = Lists.newArrayList();
        this.predictedHeadsAndDeps = Lists.newArrayList();
        for (int child = 0; child <= size; child++) {
            headsAndDeps.add(new LabeledHead());
            predictedHeadsAndDeps.add(new LabeledHead());
        }
        children = HashMultimap.create();
    }

    public void addNode(int child, int head, String depRel, int pHead, String pDepRel) {
        headsAndDeps.set(child, new LabeledHead(head, depRel));
        children.put(head, child);
        predictedHeadsAndDeps.set(child, new LabeledHead(pHead, pDepRel));
    }

    public List<LabeledHead> getAllNodes() {
        return headsAndDeps;
    }

    public int getHead(int child) {
        return headsAndDeps.get(child).headId;
    }

    public String getHeadDepRel(int child) {
        return headsAndDeps.get(child).label;
    }

    public int getPredictedHead(int child) {
        return predictedHeadsAndDeps.get(child).headId;
    }

    public String getPredictedHeadDepRel(int child) {
        return predictedHeadsAndDeps.get(child).label;
    }

    public Collection<Integer> getChildren(int head) {
        if (children.containsKey(head)) {
            return children.get(head);
        }
        return null;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((headsAndDeps == null) ? 0 : headsAndDeps.hashCode());
        result = prime * result + size;
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
        DepTree other = (DepTree) obj;

        if (size != other.size)
            return false;

        for (LabeledHead lh : headsAndDeps) {
            if (other.headsAndDeps.contains(lh) == false) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        int pos = 0;
        for (LabeledHead lh : headsAndDeps) {

            builder.append(lh.headId);
            builder.append("-");
            builder.append(lh.label);
            builder.append("->");
            builder.append(pos);
            builder.append("\n");
            pos++;
        }
        return builder.toString();
    }

    public String toString(Conll conll) {
        StringBuilder builder = new StringBuilder();
        int pos = 0;
        for (LabeledHead lh : headsAndDeps) {
            if (lh.headId != -1) {
                builder.append(conll.elements.get(lh.headId).form);
                builder.append("-");
                builder.append(lh.label);
                builder.append("->");
                builder.append(conll.elements.get(pos).form);
                builder.append("\n");
            }
            pos++;
        }
        return builder.toString();
    }
}
