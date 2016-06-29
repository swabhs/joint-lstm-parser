package edu.cmu.cs.lti.oracles.jointsynsem;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import edu.cmu.cs.lti.oracles.datastructs.Conll;
import edu.cmu.cs.lti.oracles.datastructs.ConllFileIO;
import edu.cmu.cs.lti.oracles.datastructs.DepTree;
import edu.cmu.cs.lti.oracles.datastructs.SemTree;
import edu.cmu.cs.lti.oracles.datastructs.SynSemAnalysis;

public class DataAnalyzer {

    @Parameter(names = "-inp", description = "conll file in 2008/09 format")
    // public static String input = "proj.ja.train.conll09";
    public static String input = "proj.en.conll2009.dev.conll";

    @Parameter(names = "-pred", description = "conll file in 2008/09 format")
    public static String predictions = "predicted.plem_ppos.dev.joint.conll";

    private static FreqSet predicateSenses;

    public static void pickLoriExamples() {
        ConllFileIO reader = new ConllFileIO();
        ImmutableList<Conll> inpConlls = reader.readConllFile(input, false);
        ImmutableList<Conll> predConlls = reader.readConllFile(predictions, false);

        List<Integer> selected = Arrays.asList(133, 119, 1, 102, 67);

        for (int i : selected) {
            // SynSemAnalysis inpAnalysis = new SynSemAnalysis(inpConlls.get(i));
            // SynSemAnalysis predAnalysis = new SynSemAnalysis(predConlls.get(i));
            // if (compareAnalyses(inpAnalysis, predAnalysis)) {
            // System.err.println("ex = " + i + " size = " +
            // inpAnalysis.size());
            // if (inpAnalysis.size() == 16) {
            // System.out.println(inpConlls.get(i).toString());
            System.out.println(predConlls.get(i).toString());
            // break;
            // }
            // }
        }
    }
    public static boolean compareAnalyses(SynSemAnalysis gold, SynSemAnalysis pred) {
        DepTree goldTree = gold.getDepTree();
        DepTree predTree = pred.getDepTree();
        SemTree goldSemTree = gold.getSemTree();
        SemTree predSemTree = pred.getSemTree();

        return goldSemTree.equals(predSemTree) && (goldTree.equals(predTree) == false);
    }

    public static void analyseData() {
        predicateSenses = new FreqSet();

        ConllFileIO reader = new ConllFileIO();
        ImmutableList<Conll> conlls = reader.readConllFile(input, false);

        List<SynSemAnalysis> goldAnalyses = Lists.newArrayList();

        int ex = 0;
        int numSelfArcs = 0;
        int numTotalArcs = 0;
        int numPreds = 0;
        double fracToksWhichArePreds = 0.0;

        for (Conll conll : conlls) {
            if (ex >= 0) { // our example is sent#34054 in train
                SynSemAnalysis analysis = new SynSemAnalysis(conll);
                goldAnalyses.add(analysis);
                fracToksWhichArePreds += (analysis.getSemTree().getPredicates().size() * 1.0 / analysis
                        .size());
                predicateSenses.addAll(analysis.getSemTree().getPredicates().values());
                // System.err.println(ex + " size = " + analysis.size() + " #preds = "
                // + analysis.getSemTree().getPredicates().size());

                numSelfArcs += analysis.selfArcs;
                numTotalArcs += analysis.semArcs;

            }
            ++ex;
        }
        NumberFormat formatter = new DecimalFormat("#0.00");

        System.err.println("Num sentences in corpus = " + ex);
        System.err.println("num self arcs = " + numSelfArcs);
        System.err.println("num tot arcs = " + numTotalArcs);
        System.err.println("Avg # preds in a sent = " + fracToksWhichArePreds / ex);

        System.err.println("Total number of predicate senses = " + predicateSenses.size());

        for (int f = 1; f < 100; ++f) {
            System.err
                    .println("freq " + f + " predicates = "
                            + predicateSenses.getNumberElementsWithFreq(f));
        }

        System.err.println("% self arcs = " + formatter.format(numSelfArcs * 100.0 / numTotalArcs));
        // OracleRunner.main(args);
    }

    public static void main(String[] args) {
        new JCommander(new DataAnalyzer(), args);
        // analyseData();
        pickLoriExamples();
    }

    static class FreqSet {

        private Map<String, Integer> frequencies;

        public FreqSet() {
            frequencies = Maps.newTreeMap();
        }

        public void addAll(Collection<String> col) {
            for (String x : col) {
                if (frequencies.containsKey(x)) {
                    int f = frequencies.get(x);
                    frequencies.put(x, f + 1);
                } else {
                    frequencies.put(x, 1);
                }
            }
        }

        public int getNumberElementsWithFreq(int f) {
            int result = 0;
            for (Map.Entry<String, Integer> entry : frequencies.entrySet()) {
                if (entry.getValue() == f) {
                    result += 1;
                }
            }
            return result;
        }

        public int size() {
            return frequencies.size();
        }
    }

}
