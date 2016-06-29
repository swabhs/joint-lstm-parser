package edu.cmu.cs.lti.oracles.jointsynsem;

import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayDeque;
import java.util.Iterator;
import java.util.Map;
import java.util.Stack;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;

import edu.cmu.cs.lti.oracles.datastructs.LabeledHead;
import edu.cmu.cs.lti.oracles.datastructs.SemHead;
import edu.cmu.cs.lti.oracles.datastructs.SynSemAnalysis;

public class State {

    public Stack<Integer> synStack;
    public Stack<Integer> semStack;
    public ArrayDeque<Integer> buffer;
    public Map<Integer, LabeledHead> labeledArcs;
    public Map<Integer, String> preds;
    public Multimap<Integer, SemHead> labeledSemArcs;
    public Action prevAction;

    public static PrintStream ps;

    private boolean semSwitch = false;

    private SynSemAnalysis analysis;

    public State(SynSemAnalysis analysis) {
        this.analysis = analysis;

        synStack = new Stack<Integer>();
        semStack = new Stack<Integer>();
        buffer = new ArrayDeque<Integer>(analysis.size());

        for (int i = 1; i <= analysis.size(); i++) {
            buffer.addLast(i);
        }
        buffer.addLast(SynSemAnalysis.ROOT);

        labeledArcs = Maps.newHashMap();
        labeledArcs.put(SynSemAnalysis.ROOT, new LabeledHead());

        preds = Maps.newTreeMap();
        labeledSemArcs = HashMultimap.create();

        try {
            ps = new PrintStream(System.out, true, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }

    public void setAfterSyn(
            Stack<Integer> synStack,
            ArrayDeque<Integer> buffer,
            Map<Integer, LabeledHead> labeledArcs,
            boolean semSwitch) {
        this.synStack = synStack;
        this.buffer = buffer;
        this.labeledArcs = labeledArcs;
        this.semSwitch = semSwitch;
    }

    public void setAfterSem(
            Stack<Integer> semStack,
            ArrayDeque<Integer> buffer,
            Map<Integer, String> preds,
            Multimap<Integer, SemHead> labeledSemArcs,
            boolean semSwitch) {
        this.semStack = semStack;
        this.buffer = buffer;
        this.preds = preds;
        this.labeledSemArcs = labeledSemArcs;
        this.semSwitch = semSwitch;
    }

    public boolean isSemSwitchOn() {
        return semSwitch;
    }

    private String bufferToString(ArrayDeque<Integer> buffer) {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        Iterator<Integer> it = buffer.iterator();
        while (it.hasNext()) {
            builder.append(OracleHelper.tokenPos(analysis, it.next()) + ", ");
        }
        builder.append("]");
        return builder.toString();
    }

    private String stackToString(Stack<Integer> stack) {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        Iterator<Integer> it = stack.iterator();
        while (it.hasNext()) {
            builder.append(OracleHelper.tokenPos(analysis, it.next()) + ", ");
        }
        builder.append("]");
        // builder.append("]<-top");
        return builder.toString();
    }

    private String toString(String prevAction) {
        StringBuilder builder = new StringBuilder();
        if (OracleRunner.debugModeOn) {
            // builder.append("After ");
            builder.append("\\textsc{");
        }

        builder.append(prevAction);
        if (OracleRunner.debugModeOn) {
            builder.append("}& ");
        }
        // builder.append("\n");
        if (OracleRunner.debugModeOn) {
            // builder.append("\tsyn ");
            builder.append(stackToString(synStack));
            builder.append("& ");
            // builder.append("\n\tsem ");
            builder.append(stackToString(semStack));
            builder.append("& ");
            // builder.append("\n\tbuf -> ");
            builder.append(bufferToString(buffer));
            builder.append(" \\\\");
        }
        builder.append("\n");
        return builder.toString();
    }

    public void print(String prevAction) {
        if (OracleRunner.printStates) {
            ps.print(toString(prevAction));
        }
    }

    public void print(Map<Integer, String> goldPreds) {
        if (OracleRunner.printStates) {
            System.out.println();
            for (int i = 1; i <= analysis.size(); i++) {
                String tokenPosMaybeLemma;
                if (OracleRunner.conll08) {
                    tokenPosMaybeLemma = OracleHelper.tokenPosLemma(analysis, i);
                } else {
                    tokenPosMaybeLemma = OracleHelper.tokenPos(analysis, i);
                    if (goldPreds.containsKey(i)) {
                        tokenPosMaybeLemma += "~" + analysis.lemmas.get(i);// goldPreds.get(i);
                    }
                }
                ps.print(tokenPosMaybeLemma + ", ");
            }
            ps.println(OracleHelper.tokenPos(analysis, 0));
        }
    }
}
