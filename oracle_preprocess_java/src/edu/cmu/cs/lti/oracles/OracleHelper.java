package edu.cmu.cs.lti.oracles.jointsynsem;

import java.util.List;
import java.util.Map;
import java.util.Stack;

import com.google.common.collect.Multimap;

import edu.cmu.cs.lti.oracles.datastructs.LabeledHead;
import edu.cmu.cs.lti.oracles.datastructs.SemHead;
import edu.cmu.cs.lti.oracles.datastructs.SemTree;
import edu.cmu.cs.lti.oracles.datastructs.SynSemAnalysis;

public class OracleHelper {

    public static boolean isArcInMap(Multimap<Integer, SemHead> semArcs, int pred, int arg) {
        if (semArcs.containsKey(arg) == false) {
            return false;
        }
        for (SemHead sh : semArcs.get(arg)) {
            if (sh.semHeadId == pred) {
                return true;
            }
        }
        return false;
    }

    // swap only if the lexicographic order of next stack top is higher than stack top
    public static boolean shouldWeSwap(
            Stack<Integer> semStack,
            SemTree tree,
            int top,
            int nextTop,
            int bufferFront) {
        if (semStack.size() < 2) {
            return false;
        }
        List<Integer> topConns = tree.getAllConnections(top);
        List<Integer> nextTopConns = tree.getAllConnections(nextTop);
        while (topConns.size() > 0) {
            if (topConns.get(0) > bufferFront) {
                break;
            }
            topConns = topConns.subList(1, topConns.size());
        }

        while (nextTopConns.size() > 0) {
            if (nextTopConns.get(0) >= bufferFront) {
                break;
            }
            nextTopConns = nextTopConns.subList(1, nextTopConns.size());
        }
        return compareLists(topConns, nextTopConns);
    }

    static boolean compareLists(List<Integer> topHeads, List<Integer> nextTopHeads) {
        if (topHeads.size() == 0 || nextTopHeads.size() == 0) {
            return false; // should not happen
        }
        if (topHeads.get(0) < nextTopHeads.get(0)) {
            return false;
        } else if (topHeads.get(0) == nextTopHeads.get(0)) {
            if (topHeads.size() < nextTopHeads.size()) {
                return false;
            } else if (topHeads.size() > nextTopHeads.size()) {
                return true;
            }
            return compareLists(topHeads.subList(1, topHeads.size()),
                    nextTopHeads.subList(1, nextTopHeads.size()));
        }
        return true;
    }

    public static int nextTop(Stack<Integer> stack) {
        if (stack.size() <= 1) {
            return Oracle.DEFAULTLEFTCHILD;
        }
        return stack.elementAt(stack.size() - 2);
    }

    public static Stack<Integer> swapStack(Stack<Integer> stack) {
        int firstTop = stack.pop();
        int secondTop = stack.pop();
        stack.push(firstTop);
        stack.push(secondTop);
        return stack;
    }

    static boolean checkIfCorrect(State finalState, SynSemAnalysis gold) {
        // syntactic check
        Map<Integer, LabeledHead> labeledArcs = finalState.labeledArcs;
        int pos = 0;
        for (LabeledHead lh : gold.getDepTree().getAllNodes()) {
            if (labeledArcs.containsKey(pos) == false)
                return false;
            if (labeledArcs.get(pos).equals(lh) == false) {
                return false;
            }
            pos++;
        }

        // semantic check
        SemTree goldSemTree = gold.getSemTree();
        Multimap<Integer, SemHead> predictedArcs = finalState.labeledSemArcs;
        Map<Integer, String> predictedPreds = finalState.preds;

        for (int predPos : goldSemTree.getPredicates().keySet()) {
            if (predictedPreds.containsKey(predPos) == false) {
                return false;
            }
        }
        Multimap<Integer, SemHead> goldArgs = goldSemTree.getAllArguments();
        if (predictedArcs.keySet().size() != goldArgs.keySet().size()) {
            return false;
        }
        for (int arg : goldArgs.keySet()) {
            if (!predictedArcs.containsKey(arg)) {
                return false;
            }
            for (SemHead sh : goldArgs.get(arg)) {
                if (predictedArcs.get(arg).contains(sh) == false) {
                    return false;
                }
            }
        }
        return true;
    }

    public static String tokenPos(SynSemAnalysis analysis, int pos) {
        if (OracleRunner.debugModeOn) {
            return analysis.sentence.get(pos);
        }
        return analysis.sentence.get(pos) + "-" + analysis.postags.get(pos);
    }

    public static String tokenPosLemma(SynSemAnalysis analysis, int pos) {
        return analysis.sentence.get(pos) + "-" + analysis.postags.get(pos) + "~"
                + analysis.lemmas.get(pos);
    }
}
