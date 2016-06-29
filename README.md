# lstm-parser
Transition-based joint syntactic dependency parser and semantic role labeler using stack LSTM RNN architecture

#### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)
 * [CMake](http://www.cmake.org/)

#### Checking out the project for the first time

The first time you clone the repository, you need to sync the `cnn/` submodule.

    git submodule init
    git submodule update

#### Build instructions

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2

#### Train a parsing model

The data must be in the [CoNLL 2009 format](https://ufal.mff.cuni.cz/conll2009-st/task-description.html). For best performance, it is suggested to [projectivize the treebank data](http://www.maltparser.org/userguide.html#singlemalt_info). As a preprocessing step, first convert the treebank data (in CoNLL 2009 format) into transitions, \*.transitions (this is a format usable by the joint parser), using the following commands.

    java -jar ../jointOracle.jar -inp train.conll -lemmas true > train.transitions
    java -jar ../jointOracle.jar -inp dev.conll > dev.transitions

Note that it is required to set the option ``lemmas'' to true for the training data, so that an auxilliary file, train.conll.pb.lemmas is generated, which saves the lemmas of all the predicate words. The joint parser can now run on the generated files.

    parser/lstm-parse -T train.transitions -d dev.transitions -w sskip.100.vectors --propbank_lemmas train.conll.pb.lemmas -g dev.conll -e ../eval09.pl -s dev.predictions.conll --out_model joint.model -t
    
Link to the word vectors that we used in the ACL 2015 paper for English: [sskip.100.vectors](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing).
The evaluation script, eval09.pl, is provided in the CoNLL 2009 webpage.
The model is written to the current directory.

Note-1: you can also run it without word embeddings by removing the -w option for both training and parsing.

Note-2: the training process should be stopped when the development result does not substantially improve anymore.

#### Parse data with your parsing model

The test.conll file must also be in to the [CoNLL 2009 data format](https://ufal.mff.cuni.cz/conll2009-st/task-description.html) .

    java -jar ../jointOracle.jar -inp test.conll > test.transitions

    parser/lstm-parse -T train.transitions -d test.transitions -w sskip.100.vectors --propbank_lemmas train.conll.pb.lemmas -m joint.model -s test.predictions.conll -g test.conll -e ../eval09.pl 

The parser will output a file test.predictions.conll with predicted syntax and SRL dependencies.

#### Contact

For questions and usage issues, please contact swabha@cs.cmu.edu

