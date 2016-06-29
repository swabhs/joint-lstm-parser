package edu.cmu.cs.lti.oracles.datastructs;

import java.util.List;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;

import edu.cmu.cs.lti.nlp.swabha.fileutils.BasicFileReader;
import edu.cmu.cs.lti.nlp.swabha.fileutils.BasicFileWriter;

public class ConllFileIO {

    public static String joinTabSeparatedStrings(String... strings) {
        StringBuilder builder = new StringBuilder();
        for (String str : strings) {
            builder.append(str);
            builder.append("\t");
        }
        return builder.toString();
    }

    public ImmutableList<Conll> readConllFile(String fileName, boolean is2008) {
        List<String> lines = BasicFileReader.readFile(fileName);
        ImmutableList.Builder<Conll> builder = new ImmutableList.Builder<Conll>();

        List<String> conllLines = Lists.newArrayList();
        for (String line : lines) {
            if (line.trim().equals("")) {
                if (is2008) {
                    builder.add(new Conll2008(conllLines));
                } else {
                    builder.add(new Conll2009(conllLines));
                }
                conllLines = Lists.newArrayList();
                continue;
            }
            conllLines.add(line);
        }
        return builder.build();
    }

    public void writeConllFile(String fileName, List<Conll> conlls) {
        List<String> lines = Lists.newArrayList();
        for (Conll conll : conlls) {
            lines.add(conll.toString());
        }
        BasicFileWriter.writeStrings(lines, fileName);
    }

}
