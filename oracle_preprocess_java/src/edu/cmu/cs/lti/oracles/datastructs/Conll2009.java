package edu.cmu.cs.lti.oracles.datastructs;

import java.util.List;

import com.google.common.collect.Lists;

public class Conll2009 extends Conll {

    public int correctPOS = 0;
    public int correctlemm = 0;
    public int total = 0;

    public class Conll2009Element extends ConllElement {

        public Conll2009Element(String line) {
            String[] ele = line.trim().split("\t");
            this.id = Integer.parseInt(ele[0]);
            this.form = ele[1];
            this.predictedLemma = ele[3];
            total++;
            if (ele[2].equals(ele[3])) {
                correctlemm++;
            }
            this.predictedPOS = ele[5];
            if (ele[4].equals(ele[5])) {
                correctPOS++;
            }
            // TODO: add code for feat?
            this.head = Integer.parseInt(ele[8]);
            this.predictedHead = Integer.parseInt(ele[9]);
            this.depRel = ele[10];
            this.predictedDepRel = ele[11];

            this.isPred = ele[12].equals("Y");
            this.pred = ele[13];
            this.apreds = Lists.newArrayList();
            for (int i = 14; i < ele.length; i++) {
                apreds.add(ele[i]);
            }
        }

        @Override
        public String toString() {
            StringBuilder builder = new StringBuilder();
            String fillPred = isPred ? "Y" : "_";
            String ele[] = new String[]{id + "", form, predictedLemma, predictedLemma,
                    predictedPOS,
                    predictedPOS, "_", "_",
                    head + "", head + "", depRel, depRel, fillPred, pred};
            String firstPart = ConllFileIO.joinTabSeparatedStrings(ele);
            builder.append(firstPart);
            for (int i = 0; i < apreds.size(); i++) {
                builder.append(apreds.get(i));
                if (i < apreds.size() - 1) {
                    builder.append("\t");
                }
            }
            builder.append("\n");
            return builder.toString();
        }

    }

    public Conll2009(List<String> lines) {
        this.elements = Lists.newArrayList();
        for (String line : lines) {
            elements.add(new Conll2009Element(line));
        }
        this.size = lines.size();
    }

}
