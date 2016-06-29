package edu.cmu.cs.lti.oracles.datastructs;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;

import edu.cmu.cs.lti.oracles.jointsynsem.Oracle;

public class SemTree {

    private Map<Integer, SemStructure> semStructs;
    private Map<Integer, String> predicates;
    private Multimap<Integer, SemHead> arguments;
    private Map<Integer, Integer> rightMostChild;

    public SemTree() {
        this.semStructs = Maps.newTreeMap();
        this.predicates = Maps.newTreeMap();
        this.arguments = TreeMultimap.create();
        this.rightMostChild = Maps.newTreeMap();
    }

    public void addSemStructure(int pos, String pred) {
        semStructs.put(pos, new SemStructure(pos, pred));
        predicates.put(pos, pred);
        rightMostChild.put(pos, Oracle.DEFAULTLEFTCHILD); // there are predicates without args
    }

    public void addArgument(int predPos, int argPos, String argLabel) {
        SemStructure struct = semStructs.get(predPos);
        struct.addArg(argPos, argLabel);
        semStructs.put(predPos, struct);
        arguments.put(argPos, new SemHead(predPos, argLabel));
        rightMostChild.put(predPos, argPos); // because you always add arguments left to right
    }

    public Map<Integer, String> getPredicates() {
        return predicates;
    }

    public Multimap<Integer, SemHead> getAllArguments() {
        return arguments;
    }

    public int getRightMostChild(int predPos) {
        if (rightMostChild.containsKey(predPos) == false) {
            System.err.println("no right child for " + predPos);
        }
        return rightMostChild.get(predPos);
    }

    public int getSemHeadsCount(int argPos) {
        if (arguments.containsKey(argPos) == false) {
            return 0;
        }
        return arguments.get(argPos).size();
    }

    // returns sem head if it exists
    public SemHead getSemHead(int pred, int argPosition) {
        if (semStructs.containsKey(pred)) {
            if (semStructs.get(pred).isArg(argPosition)) {
                return semStructs.get(pred).getSemHead(argPosition);
            }
        }
        throw new IllegalArgumentException("no arc between " + pred + " and " + argPosition);
    }

    public Set<Integer> getAllChildren(int pred) {
        if (predicates.containsKey(pred)) {
            return semStructs.get(pred).getAllArgs().keySet();
        }
        return new HashSet<Integer>();
    }

    public List<Integer> getAllConnections(int pos) {
        List<Integer> connections = Lists.newArrayList();
        if (arguments.containsKey(pos)) {
            for (SemHead sh : arguments.get(pos)) {
                connections.add(sh.semHeadId);
            }
        }
        if (isPred(pos)) {
            connections.addAll(getAllChildren(pos));
        }
        Collections.sort(connections);
        return connections;
    }

    public boolean containsArc(int semHeadId, int argPos) {
        if (arguments.containsKey(argPos) == false) {
            return false;
        }
        for (SemHead sh : arguments.get(argPos)) {
            if (sh.semHeadId == semHeadId) {
                return true;
            }
        }
        return false;
    }

    public boolean isArg(int pos) {
        return arguments.containsKey(pos);
    }

    public boolean isPred(int pos) {
        return semStructs.containsKey(pos);
    }

    public boolean isNeitherPredNorArg(int pos) {
        return (semStructs.containsKey(pos) == false && arguments.containsKey(pos) == false);
    }

    public boolean isBothPredArg(int pos) {
        return isArg(pos) && isPred(pos);
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((semStructs == null) ? 0 : semStructs.hashCode());
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
        SemTree other = (SemTree) obj;

        for (Map.Entry<Integer, SemStructure> entry : semStructs.entrySet()) {
            // System.err.println("predicate at " + entry.getKey());
            if (other.semStructs.containsKey(entry.getKey()) == false) {
                // System.err.println("predicate not found");
                return false;
            }
            if (entry.getValue().equals(other.semStructs.get(entry.getKey())) == false) {
                // System.err.println("argument structure unequal");
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (int pred : semStructs.keySet()) {
            builder.append("PRED-" + pred + ":" + semStructs.get(pred).getPred() + "\n");
            for (int arg : semStructs.get(pred).getAllArgs().keySet()) {
                SemHead semHead = semStructs.get(pred).getSemHead(arg);
                builder.append(pred);
                builder.append("-");
                builder.append(semHead.semLabel);
                builder.append("->");
                builder.append(arg);
                builder.append("\n");
            }
        }
        return builder.toString();
    }

    public String toString(Conll conll) {
        StringBuilder builder = new StringBuilder();
        for (int pred : semStructs.keySet()) {
            builder.append("PRED-" + pred + ":" + semStructs.get(pred).getPred() + "\n");
            for (int arg : semStructs.get(pred).getAllArgs().keySet()) {
                SemHead semHead = semStructs.get(pred).getSemHead(arg);
                builder.append(conll.elements.get(pred - 1).form);
                builder.append("-");
                builder.append(semHead.semLabel);
                builder.append("->");
                builder.append(conll.elements.get(arg - 1).form);
                builder.append("\n");
            }
        }
        return builder.toString();
    }
}
