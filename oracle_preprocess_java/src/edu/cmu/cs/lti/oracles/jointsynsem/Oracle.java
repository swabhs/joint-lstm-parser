package edu.cmu.cs.lti.oracles.jointsynsem;

import java.util.ArrayDeque;
import java.util.Map;
import java.util.Stack;

import com.google.common.collect.Multimap;

import edu.cmu.cs.lti.oracles.datastructs.DepTree;
import edu.cmu.cs.lti.oracles.datastructs.LabeledHead;
import edu.cmu.cs.lti.oracles.datastructs.SemHead;
import edu.cmu.cs.lti.oracles.datastructs.SemTree;
import edu.cmu.cs.lti.oracles.datastructs.SynSemAnalysis;

public class Oracle {

    private State state;

    public static final int DEFAULTLEFTCHILD = -30;
    protected boolean labeled = true;

    public Oracle(State state) {
        this.state = state;
    }

    private Transition getOracleSyntacticTransition(SynSemAnalysis gold) {
        ArrayDeque<Integer> buffer = state.buffer;
        Transition transition = new Transition();
        Stack<Integer> synStack = state.synStack;
        Map<Integer, LabeledHead> labeledArcs = state.labeledArcs;

        boolean switchToSem = false;

        if (synStack.empty()) { // SHIFT
            synStack.push(buffer.peek()); // buffer.removeFirst());
            state.setAfterSyn(synStack, buffer, state.labeledArcs, true);
            return new Transition(Action.SHIFT, null);
        }

        DepTree goldTree = gold.getDepTree();
        int stackTopId = synStack.peek();
        int stackTopHead = goldTree.getHead(stackTopId);

        int bufferFrontId = buffer.peek();
        int bufferFrontHead = goldTree.getHead(bufferFrontId);

        int leftMostChildOfBufferFront = gold.getLeftmostChild(bufferFrontId);

        if (bufferFrontId == stackTopHead) {
            // LEFT ARC
            labeledArcs.put(stackTopId,
                    new LabeledHead(bufferFrontId, goldTree.getHeadDepRel(stackTopId)));
            synStack.pop();
            transition = labeled
                    ? new Transition(Action.LA, goldTree.getHeadDepRel(stackTopId))
                    : new Transition(Action.LA, null);
        } else if (stackTopId == bufferFrontHead) {
            // RIGHT ARC
            labeledArcs.put(bufferFrontId,
                    new LabeledHead(stackTopId, goldTree.getHeadDepRel(bufferFrontId)));
            synStack.push(buffer.peek());// removeFirst()
            switchToSem = true;
            transition = labeled
                    ? new Transition(Action.RA, goldTree.getHeadDepRel(bufferFrontId))
                    : new Transition(Action.RA, null);
        } else if (stackTopHead != 0
                && ((bufferFrontHead != 0 && bufferFrontHead < stackTopId)
                || (leftMostChildOfBufferFront != Oracle.DEFAULTLEFTCHILD
                && leftMostChildOfBufferFront < stackTopId))) {
            // REDUCE
            synStack.pop();
            transition = new Transition(Action.REDUCE, null);
        } else {
            // SHIFT
            synStack.push(buffer.peek()); // buffer.removeFirst());
            switchToSem = true;
            transition = new Transition(Action.SHIFT, null);
        }
        state.setAfterSyn(synStack, buffer, labeledArcs, switchToSem);
        return transition;
    }

    private Transition getOracleSemanticTransition(SynSemAnalysis gold) {
        ArrayDeque<Integer> buffer = state.buffer;
        Transition transition = new Transition();
        Stack<Integer> semStack = state.semStack;
        Map<Integer, String> preds = state.preds;
        Multimap<Integer, SemHead> semArcs = state.labeledSemArcs;

        boolean keepInSem = true;

        SemTree goldTree = gold.getSemTree();
        Map<Integer, String> goldPreds = goldTree.getPredicates();

        if (goldPreds.containsKey(buffer.peek()) && !preds.containsKey(buffer.peek())) {// 1st PRED
            preds.put(buffer.peek(), goldPreds.get(buffer.peek()));
            state.setAfterSem(semStack, buffer, preds, semArcs, keepInSem);
            state.prevAction = Action.PRED;
            return new Transition(Action.PRED, goldPreds.get(buffer.peek()));
        }

        if (goldTree.containsArc(buffer.peek(), buffer.peek())
                && OracleHelper.isArcInMap(semArcs, buffer.peek(), buffer.peek()) == false) {
            SemHead sh = goldTree.getSemHead(buffer.peek(), buffer.peek());
            semArcs.put(buffer.peek(), sh);
            return new Transition(Action.SELF, sh.semLabel);
        }

        if (semStack.empty()) { // SEMSHIFT
            semStack.push(buffer.removeFirst());
            state.setAfterSem(semStack, buffer, preds, semArcs, !keepInSem);
            state.prevAction = Action.SEMSHIFT;
            return new Transition(Action.SEMSHIFT, null);
        }

        int stackTopId = semStack.peek();
        int nextStackTop = OracleHelper.nextTop(semStack);
        int bufferFrontId = buffer.peek();

        // special case: self-referential arcs
        // int[] splCases = new int[]{stackTopId, bufferFrontId};
        // for (int splCase : splCases) {
        // if (goldTree.containsArc(splCase, splCase)
        // && OracleHelper.isArcInMap(semArcs, splCase, splCase) == false) {
        // SemHead sh = goldTree.getSemHead(splCase, splCase);
        // semArcs.put(splCase, sh);
        // }
        // }

        if (goldTree.containsArc(bufferFrontId, bufferFrontId)
                && OracleHelper.isArcInMap(semArcs, bufferFrontId, bufferFrontId) == false) {
            SemHead sh = goldTree.getSemHead(bufferFrontId, bufferFrontId);
            semArcs.put(bufferFrontId, sh);
            transition = new Transition(Action.SELF, sh.semLabel);
        } else if (goldPreds.containsKey(bufferFrontId) && !preds.containsKey(bufferFrontId)) {
            // PRED
            preds.put(bufferFrontId, goldPreds.get(bufferFrontId));
            transition = new Transition(Action.PRED, goldPreds.get(bufferFrontId));
        } else if (goldTree.containsArc(bufferFrontId, stackTopId)
                && OracleHelper.isArcInMap(semArcs, bufferFrontId, stackTopId) == false) {
            // SEM LEFT ARC
            SemHead sh = goldTree.getSemHead(bufferFrontId, stackTopId);
            semArcs.put(stackTopId, sh);
            transition = new Transition(Action.SEMLA, sh.semLabel);
        } else if (goldTree.containsArc(stackTopId, bufferFrontId)
                && OracleHelper.isArcInMap(semArcs, stackTopId, bufferFrontId) == false) {
            // SEM RIGHT ARC
            SemHead sh = goldTree.getSemHead(stackTopId, bufferFrontId);
            semArcs.put(bufferFrontId, sh);
            transition = new Transition(Action.SEMRA, sh.semLabel);
        } else if (goldTree.isNeitherPredNorArg(stackTopId)
                || (goldTree.isArg(stackTopId) && goldTree.isPred(stackTopId) == false && goldTree
                        .getSemHeadsCount(stackTopId) == semArcs.get(stackTopId).size())
                || (goldTree.isPred(stackTopId) && goldTree.isArg(stackTopId) == false && goldTree
                        .getRightMostChild(stackTopId) <= bufferFrontId)
                || (goldTree.isBothPredArg(stackTopId) && goldTree
                        .getSemHeadsCount(stackTopId) == semArcs.get(stackTopId).size() && goldTree
                        .getRightMostChild(stackTopId) <= bufferFrontId)
                || bufferFrontId == SynSemAnalysis.ROOT) {
            // SEM REDUCE
            // the stack top is a pred and all its children have been seen
            // or the stack top is an arg and all its parents have been seen
            // or its neither a parent nor a child) {
            semStack.pop();
            transition = new Transition(Action.SEMREDUCE, null);
        } else if (state.prevAction != Action.SWAP
                && OracleHelper.shouldWeSwap(semStack, goldTree, stackTopId, nextStackTop,
                        bufferFrontId)) {
            // && ((goldTree.containsArc(nextStackTop, bufferFrontId)
            // && isArcInMap(semArcs, nextStackTop, bufferFrontId) == false)
            // || (goldTree.containsArc(bufferFrontId, nextStackTop)
            // && isArcInMap(semArcs, bufferFrontId, nextStackTop) == false))) {
            // SWAP
            semStack = OracleHelper.swapStack(semStack);
            transition = new Transition(Action.SWAP, null);
        } else {
            // SEM SHIFT
            semStack.push(buffer.removeFirst());
            transition = new Transition(Action.SEMSHIFT, null);
            keepInSem = false;
        }
        state.setAfterSem(semStack, buffer, preds, semArcs, keepInSem);
        state.prevAction = transition.action;
        return transition;
    }

    public State getAllTransitions(SynSemAnalysis goldAnalysis) {
        int t = 0;
        state.print(goldAnalysis.getSemTree().getPredicates());
        while (!state.buffer.isEmpty()) {
            Transition transition = new Transition();
            if (state.isSemSwitchOn()) {
                transition = getOracleSemanticTransition(goldAnalysis);
            } else {
                transition = getOracleSyntacticTransition(goldAnalysis);
            }

            switch (transition.action) {
                case LA :
                    state.print("LA(" + transition.label + ")");
                    break;
                case RA :
                    state.print("RA(" + transition.label + ")");
                    break;
                case SHIFT :
                    state.print("SH");
                    break;
                case REDUCE :
                    state.print("RE");
                    break;
                case SEMLA :
                    state.print("SL(" + transition.label + ")");
                    break;
                case SEMRA :
                    state.print("SR(" + transition.label + ")");
                    break;
                case SELF :
                    state.print("SE(" + transition.label + ")");
                    break;
                case SEMREDUCE :
                    state.print("MR");
                    break;
                case SEMSHIFT :
                    state.print("SS");
                    break;
                case PRED :
                    // revert -- temp fix. TODO
                    state.print("PR(" + transition.label + ")");
                    break;
                case SWAP :
                    state.print("SW");
                    break;
                default :
                    break;
            }
            ++t;
        }
        return state;
    }
}
