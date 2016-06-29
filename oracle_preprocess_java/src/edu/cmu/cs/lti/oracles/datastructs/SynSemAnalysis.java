package edu.cmu.cs.lti.oracles.datastructs;

import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import edu.cmu.cs.lti.oracles.jointsynsem.Oracle;

public class SynSemAnalysis {

    private DepTree depTree;
    private SemTree semTree;
    private Map<Integer, Integer> leftMostChild;

    public List<String> sentence;
    public List<String> postags;
    public List<String> lemmas;

    public static final int ROOT = 0;

    public int selfArcs = 0;
    public int semArcs = 0;

    public SynSemAnalysis(Conll conll) {
        depTree = new DepTree(conll.size);
        semTree = new SemTree();
        leftMostChild = Maps.newTreeMap();

        sentence = Lists.newArrayList();
        sentence.add("ROOT");

        postags = Lists.newArrayList();
        postags.add("NOTAG");

        lemmas = Lists.newArrayList();
        lemmas.add("NOLEM");

        // first pass, add all predicates
        List<Integer> predPositions = Lists.newArrayList();
        for (ConllElement ele : conll.elements) {
            depTree.addNode(ele.id, ele.head, ele.depRel, ele.predictedHead, ele.predictedDepRel);
            if (ele.isPred) {
                semTree.addSemStructure(ele.id, ele.pred);
                predPositions.add(ele.id);
            }
            if (leftMostChild.containsKey(ele.head) == false) {
                leftMostChild.put(ele.head, ele.id);
            }
            sentence.add(ele.form);
            postags.add(ele.predictedPOS);
            lemmas.add(ele.predictedLemma);
        }

        // second pass add all arguments
        for (int argPos = 0; argPos < conll.elements.size(); argPos++) {
            ConllElement ele = conll.elements.get(argPos);

            for (int predNum = 0; predNum < predPositions.size(); predNum++) {
                String argType = ele.apreds.get(predNum);
                if (argType.equals("_")) {
                    continue;
                }
                semTree.addArgument(predPositions.get(predNum), ele.id, argType);
                ++semArcs;
                if (predPositions.get(predNum) == ele.id) {
                    ++selfArcs;
                    // if (ele.predictedPOS.contains("NN") == true) {
                    // System.out.println(ele.predictedPOS);
                    // System.err.println(conll.toString());
                    // // System.exit(1);
                    // }
                }
            }
        }
    }

    public DepTree getDepTree() {
        return depTree;
    }

    public SemTree getSemTree() {
        return semTree;
    }

    public int size() {
        return depTree.size;
    }

    public int getLeftmostChild(int parent) {
        if (leftMostChild.containsKey(parent)) {
            return leftMostChild.get(parent);
        }
        return Oracle.DEFAULTLEFTCHILD;
    }
}
