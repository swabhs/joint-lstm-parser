package edu.cmu.cs.lti.oracles.datastructs;

import java.util.List;

import com.google.common.collect.Lists;

public class Conll2008 extends Conll {

    public class Conll2008Element extends ConllElement {

        public Conll2008Element(String line) {
            String[] ele = line.trim().split("\t");
            this.id = Integer.parseInt(ele[0]);
            this.form = ele[1];
            this.predictedLemma = ele[2];
            this.predictedPOS = ele[4];
            this.head = Integer.parseInt(ele[8]);
            this.depRel = ele[9];
            this.pred = ele[10];
            this.isPred = !pred.equals("_");
            this.apreds = Lists.newArrayList();
            for (int i = 11; i < ele.length; i++) {
                apreds.add(ele[i]);
            }
        }

        @Override
        public String toString() {
            StringBuilder builder = new StringBuilder();
            String ele[] = new String[]{id + "", form, predictedLemma, "_", predictedPOS, "_", "_",
                    "_", head + "", depRel, pred};
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

    public Conll2008(List<String> lines) {
        this.elements = Lists.newArrayList();
        for (String line : lines) {
            elements.add(new Conll2008Element(line));
        }
        this.size = lines.size();
    }
}