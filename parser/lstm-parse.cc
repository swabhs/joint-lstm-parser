#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>

#include <unordered_map>
#include <unordered_set>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include "actions.h"
#include "c2.h"

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"

cpyp::Corpus corpus;

volatile bool requested_stop = false;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;
unsigned REL_DIM = 8;
unsigned PRED_DIM = 30;

unsigned LSTM_CHAR_OUTPUT_DIM = 100; //Miguel
bool USE_SPELLING = false;
bool USE_POS = true;

constexpr const char* ROOT_SYMBOL = "ROOT";
unsigned kROOT_SYMBOL = 0;
unsigned kUNK;
unsigned UNK_STRATEGY;
double UNK_PROB;

unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned POS_SIZE = 0;
unsigned PRED_SIZE = 0;
unsigned BEAM_SIZE = 0;
unsigned CHAR_SIZE = 255; //size of ascii chars... Miguel

bool USE_DROPOUT = false;
float DROPOUT = 0.0f;

using namespace cnn::expr;
using namespace cnn;
using namespace std;
using namespace badname;
namespace po = boost::program_options;

unordered_map<unsigned, vector<float>> pretrained;

string param_fname;
string output_conll;

void init_command_line(int argc, char** argv, po::variables_map* conf) {

    po::options_description opts("Configuration options");
    opts.add_options()("training_data,T", po::value<string>(),
            "list of Transitions - training corpus")("dev_data,d",
            po::value<string>(), "Development corpus")("test_data,p",
            po::value<string>(), "Test corpus")("unk_strategy,o",
            po::value<unsigned>()->default_value(1),
            "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")(
            "unk_prob,u", po::value<double>()->default_value(0.2),
            "Probably with which to replace singletons with UNK in training data")(
            "model,m", po::value<string>(), "load saved model from this file")(
            "out_model", po::value<string>(),
            "save output model to this soft link")("use_pos_tags,P",
            "make POS tags visible to parser")("beam_size,b",
            po::value<unsigned>()->default_value(1), "beam size")("layers",
            po::value<unsigned>()->default_value(2), "number of LSTM layers")(
            "action_dim", po::value<unsigned>()->default_value(100),
            "action embedding size")("input_dim",
            po::value<unsigned>()->default_value(32), "input embedding size")(
            "hidden_dim", po::value<unsigned>()->default_value(100),
            "hidden dimension")("pretrained_dim",
            po::value<unsigned>()->default_value(100), "pretrained input dim")(
            "pos_dim", po::value<unsigned>()->default_value(12),
            "POS dimension")("rel_dim",
            po::value<unsigned>()->default_value(20), "relation dimension")(
            "pred_dim", po::value<unsigned>()->default_value(100),
            "predicate dimension")("propbank_lemmas", po::value<string>(),
            "lemmas mapped to senses in propbank")("lstm_input_dim",
            po::value<unsigned>()->default_value(100), "LSTM input dimension")(
            "dropout", po::value<float>()->default_value(0.2f), "Dropout rate")(
            "train,t", "Should training be run?")("words,w",
            po::value<string>(), "pretrained word embeddings")("use_lowerwv",
            "Lowercase tokens for wv compatibility")("use_spelling,S",
            "Use spelling model")("gold_conll,g", po::value<string>(),
            "Gold dev/test conll file for eval")("output_conll,s",
            po::value<string>(), "Predicted dev/test conll file for eval")(
            "eval_script,e", po::value<string>(),
            "CONLL 2009 evaluation script")("cnn_mem",
            po::value<unsigned>()->default_value(1024), "cnn memory")("help,h",
            "Help");
    po::options_description dcmdline_options;
    dcmdline_options.add(opts);
    po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
    if (conf->count("help")) {
        cerr << dcmdline_options << endl;
        exit(1);
    }
    if (conf->count("training_data") == 0) {
        cerr << "Please specify --traing_data (-T):"
                " this is required to determine the vocabulary mapping,"
                " even if the parser is used in prediction mode.\n";
        exit(1);
    }
}

struct ParserBuilder {
    LSTMBuilder stack_lstm;
    LSTMBuilder sem_stack_lstm;
    LSTMBuilder buffer_lstm;
    LSTMBuilder action_lstm;

    LookupParameters* p_tok; // word embeddings
    LookupParameters* p_emb; // pre-trained word embeddings (not updated)
    LookupParameters* p_pos; // POS tag embeddings
    LookupParameters* p_pred; // predicate embeddings
    LookupParameters* p_act; // input action embeddings
    LookupParameters* p_syn_label; // syntactic label embeddings
    LookupParameters* p_sem_label; // semantic label embeddings TODO(Swabha): needed?

    Parameters* p_parsact_bias; // parser state bias
    Parameters* p_act2parsact; // action LSTM to parser state
    Parameters* p_buf2parsact; // buffer LSTM to parser state
    Parameters* p_synst2parsact; // syntactic stack LSTM to parser state
    Parameters* p_semst2parsact; // semantic stack LSTM to parser state

    Parameters* p_head_comp; // head matrix for composition function
    Parameters* p_mod_comp; // dependency matrix for composition function
    Parameters* p_label_comp; // relation matrix for composition function
    Parameters* p_comp_bias; // composition function bias

    Parameters* p_head_comp2; // head matrix for semantic composition function
    Parameters* p_mod_comp2; // dependency matrix for semantic composition function
    Parameters* p_label_comp2; // relation matrix for semantic composition function
    Parameters* p_comp2_bias; // semantic composition function bias

    Parameters* p_pred_comp; // predicate matrix for predicate composition function
    Parameters* p_comp3_bias; // semantic composition function bias

    Parameters* p_tok2l; // word to LSTM input
    Parameters* p_pos2l; // POS to LSTM input
    Parameters* p_emb2l; // pre-trained word embeddings to LSTM input
    Parameters* p_inp_bias; // LSTM input bias

    Parameters* p_parse2next_act;   // parser state to action
    Parameters* p_act_bias;  // action bias

    Parameters* p_act_start;  // empty action set
    Parameters* p_buf_guard;  // end of buffer
    Parameters* p_stack_guard;  // end of stack
    Parameters* p_sem_stack_guard;  // end of stack

    // character model params by Miguel
    Parameters* p_start_of_word;  // dummy <s> symbol
    Parameters* p_end_of_word; // dummy </s> symbol
    LookupParameters* char_emb; // mapping of characters to vectors

    LSTMBuilder fw_char_lstm;
    LSTMBuilder bw_char_lstm;

    explicit ParserBuilder(Model* model,
            const unordered_map<unsigned, vector<float>>& pretrained) :
            stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model), sem_stack_lstm(
                    LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model), buffer_lstm(
                    LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model), action_lstm(
                    LAYERS, ACTION_DIM, HIDDEN_DIM, model),

            p_tok(model->add_lookup_parameters(VOCAB_SIZE, { INPUT_DIM, 1 })), p_pred(
                    model->add_lookup_parameters(PRED_SIZE, { PRED_DIM, 1 })), p_act(
                    model->add_lookup_parameters(ACTION_SIZE,
                            { ACTION_DIM, 1 })), p_syn_label(
                    model->add_lookup_parameters(ACTION_SIZE, { REL_DIM, 1 })), p_sem_label(
                    model->add_lookup_parameters(ACTION_SIZE, { REL_DIM, 1 })),

            p_parsact_bias(model->add_parameters( { HIDDEN_DIM, 1 })), p_act2parsact(
                    model->add_parameters( { HIDDEN_DIM, HIDDEN_DIM })), p_buf2parsact(
                    model->add_parameters( { HIDDEN_DIM, HIDDEN_DIM })), p_synst2parsact(
                    model->add_parameters( { HIDDEN_DIM, HIDDEN_DIM })), p_semst2parsact(
                    model->add_parameters( { HIDDEN_DIM, HIDDEN_DIM })),

            p_head_comp(
                    model->add_parameters( { LSTM_INPUT_DIM, LSTM_INPUT_DIM })), p_mod_comp(
                    model->add_parameters( { LSTM_INPUT_DIM, LSTM_INPUT_DIM })), p_label_comp(
                    model->add_parameters( { LSTM_INPUT_DIM, REL_DIM })), p_comp_bias(
                    model->add_parameters( { LSTM_INPUT_DIM, 1 })),

            p_head_comp2(
                    model->add_parameters( { LSTM_INPUT_DIM, LSTM_INPUT_DIM })), p_mod_comp2(
                    model->add_parameters( { LSTM_INPUT_DIM, LSTM_INPUT_DIM })), p_label_comp2(
                    model->add_parameters( { LSTM_INPUT_DIM, REL_DIM })), p_comp2_bias(
                    model->add_parameters( { LSTM_INPUT_DIM, 1 })), p_pred_comp(
                    model->add_parameters( { LSTM_INPUT_DIM, PRED_DIM })), p_comp3_bias(
                    model->add_parameters( { LSTM_INPUT_DIM, 1 })),

            p_tok2l(model->add_parameters( { LSTM_INPUT_DIM, INPUT_DIM })), p_inp_bias(
                    model->add_parameters( { LSTM_INPUT_DIM, 1 })),

            p_parse2next_act(
                    model->add_parameters( { ACTION_SIZE, HIDDEN_DIM })), p_act_bias(
                    model->add_parameters( { ACTION_SIZE, 1 })),

            p_act_start(model->add_parameters( { ACTION_DIM, 1 })), p_buf_guard(
                    model->add_parameters( { LSTM_INPUT_DIM, 1 })), p_stack_guard(
                    model->add_parameters( { LSTM_INPUT_DIM, 1 })), p_sem_stack_guard(
                    model->add_parameters( { LSTM_INPUT_DIM, 1 })),

            p_start_of_word(model->add_parameters( { LSTM_INPUT_DIM, 1 })), //Miguel
            p_end_of_word(model->add_parameters( { LSTM_INPUT_DIM, 1 })), //Miguel
            char_emb(model->add_lookup_parameters(CHAR_SIZE, { INPUT_DIM, 1 })), //Miguel
            fw_char_lstm(LAYERS, LSTM_INPUT_DIM, LSTM_CHAR_OUTPUT_DIM / 2,
                    model), //Miguel
            bw_char_lstm(LAYERS, LSTM_INPUT_DIM, LSTM_CHAR_OUTPUT_DIM / 2,
                    model) /*Miguel*/{

        if (USE_POS) {
            p_pos = model->add_lookup_parameters(POS_SIZE, { POS_DIM, 1 });
            p_pos2l = model->add_parameters( { LSTM_INPUT_DIM, POS_DIM });
        }
        if (pretrained.size() > 0) { // using word vectors
            p_emb = model->add_lookup_parameters(corpus.tok_dict.size() + 1, {
                    PRETRAINED_DIM, 1 });
            for (auto it : pretrained) {
                p_emb->Initialize(it.first, it.second);
            }
            p_emb2l = model->add_parameters(
                    { LSTM_INPUT_DIM, PRETRAINED_DIM });
        } else {
            p_emb = nullptr;
            p_emb2l = nullptr;
        }
    }

    // given the first character of a UTF8 block, find out how wide it is
    // see http://en.wikipedia.org/wiki/UTF-8 for more info
    inline unsigned int get_UTF8_len(unsigned char x) {
        if (x < 0x80)
            return 1;
        else if ((x >> 5) == 0x06)
            return 2;
        else if ((x >> 4) == 0x0e)
            return 3;
        else if ((x >> 3) == 0x1e)
            return 4;
        else if ((x >> 2) == 0x3e)
            return 5;
        else if ((x >> 1) == 0x7e)
            return 6;
        else
            return 0;
    }

    // *** if correct_actions is empty, this runs greedy decoding ***
    // returns parse actions for input sentence (in training just returns the reference)
    // OOV handling: raw_sent will have the actual words
    //               sent will have words replaced by appropriate UNK tokens
    // this lets us use pretrained embeddings, when available, for words that were OOV in the
    // parser training data
    JointParse log_prob_parser(ComputationGraph* hg,
            const vector<unsigned>& raw_sent, // raw sentence
            const vector<unsigned>& sent, // sent with oovs replaced
            const vector<unsigned>& pos_seq,
            const map<int, unsigned>& gold_preds,
            const vector<unsigned>& correct_actions, double *right) {

        vector<unsigned> results;
        const bool build_training_graph = correct_actions.size() > 0;

        if (USE_DROPOUT && build_training_graph) {
            stack_lstm.set_dropout(DROPOUT);
            sem_stack_lstm.set_dropout(DROPOUT);
            buffer_lstm.set_dropout(DROPOUT);
            action_lstm.set_dropout(DROPOUT);
        } else {
            stack_lstm.disable_dropout();
            sem_stack_lstm.disable_dropout();
            buffer_lstm.disable_dropout();
            action_lstm.disable_dropout();
        }

        stack_lstm.new_graph(*hg);
        sem_stack_lstm.new_graph(*hg);
        buffer_lstm.new_graph(*hg);
        action_lstm.new_graph(*hg);

        stack_lstm.start_new_sequence();
        sem_stack_lstm.start_new_sequence();
        buffer_lstm.start_new_sequence();
        action_lstm.start_new_sequence();

        // variables in the computation graph representing the parameters
        Expression parsact_bias = parameter(*hg, p_parsact_bias);

        Expression head_comp = parameter(*hg, p_head_comp);
        Expression mod_comp = parameter(*hg, p_mod_comp);
        Expression label_comp = parameter(*hg, p_label_comp);
        Expression comp_bias = parameter(*hg, p_comp_bias);

        Expression head_comp2 = parameter(*hg, p_head_comp2);
        Expression mod_comp2 = parameter(*hg, p_mod_comp2);
        Expression label_comp2 = parameter(*hg, p_label_comp2);
        Expression comp2_bias = parameter(*hg, p_comp2_bias);

        Expression pred_comp = parameter(*hg, p_pred_comp);
        Expression comp3_bias = parameter(*hg, p_comp3_bias);

        Expression synst2next_act = parameter(*hg, p_synst2parsact);
        Expression semst2next_act = parameter(*hg, p_semst2parsact);
        Expression buf2next_act = parameter(*hg, p_buf2parsact);
        Expression act2next_act = parameter(*hg, p_act2parsact);

        Expression tok2l = parameter(*hg, p_tok2l);
        Expression pos2l;
        if (USE_POS)
            pos2l = parameter(*hg, p_pos2l);
        Expression emb2l;
        if (p_emb2l)
            emb2l = parameter(*hg, p_emb2l);
        Expression inp_bias = parameter(*hg, p_inp_bias);

        Expression state2next_act = parameter(*hg, p_parse2next_act);
        Expression act_bias = parameter(*hg, p_act_bias);

        Expression act_start = parameter(*hg, p_act_start);
        action_lstm.add_input(act_start);

        vector < Expression > buffer(sent.size() + 1);
        vector<int> bufferi(sent.size() + 1); // position of the words in the sentence

        Expression word_end = parameter(*hg, p_end_of_word); //Miguel
        Expression word_start = parameter(*hg, p_start_of_word); //Miguel

        if (USE_SPELLING) {
            fw_char_lstm.new_graph(*hg);
            bw_char_lstm.new_graph(*hg);
        }

        for (unsigned i = 0; i < sent.size(); ++i) {
            //assert(sent[i] < VOCAB_SIZE);
            //Expression w = lookup(*hg, p_tok, sent[i]);

            unsigned wi = sent[i];
            string ww = corpus.tok_dict.Convert(wi);

            Expression w;
            /**********SPELLING MODEL*****************/
            if (USE_SPELLING) {
                //cout<<"using spelling"<<"\n";
                if (ww.length() == 4 && ww[0] == 'R' && ww[1] == 'O'
                        && ww[2] == 'O' && ww[3] == 'T') {
                    w = lookup(*hg, p_tok, sent[i]); //we do not need a LSTM encoding for the root word, so we put it directly-.
                } else {
                    fw_char_lstm.start_new_sequence();
                    //cerr<<"start_new_sequence done"<<"\n";

                    fw_char_lstm.add_input(word_start);
                    //cerr<<"added start of word symbol"<<"\n";
                    vector<int> strevbuffer;
                    for (unsigned j = 0; j < ww.length();
                            j += get_UTF8_len(ww[j])) {
                        //cerr<<j<<":"<<w[j]<<"\n";
                        string wj;
                        for (unsigned h = j; h < j + get_UTF8_len(ww[j]); h++) {
                            wj += ww[h];
                        }
                        //cout<<"fw"<<wj<<"\n";
                        int wjint = corpus.chars_int_map[wj];
                        //cout<<"fw:"<<wjint<<"\n";
                        strevbuffer.push_back(wjint);
                        Expression cj = lookup(*hg, char_emb, wjint);
                        fw_char_lstm.add_input(cj);
                        //cout<<"Inputdim:"<<LSTM_INPUT_DIM<<"\n";
                        //hg->incremental_forward();
                    }

                    fw_char_lstm.add_input(word_end);
                    //cerr<<"added end of word symbol"<<"\n";
                    Expression fw_i = fw_char_lstm.back();
                    //cerr<<"fw_char_lstm.back() done"<<"\n";

                    bw_char_lstm.start_new_sequence();
                    //cerr<<"bw start new sequence done"<<"\n";

                    bw_char_lstm.add_input(word_end);
                    //for (unsigned j=w.length()-1;j>=0;j--){
                    /*for (unsigned j=w.length();j-->0;){
                     //cerr<<j<<":"<<w[j]<<"\n";
                     Expression cj=lookup(*hg, char_emb, w[j]);
                     bw_char_lstm.add_input(cj);
                     }*/

                    while (!strevbuffer.empty()) {
                        int wjint = strevbuffer.back();
                        //cout<<"bw:"<<wjint<<"\n";
                        Expression cj = lookup(*hg, char_emb, wjint);
                        bw_char_lstm.add_input(cj);
                        strevbuffer.pop_back();
                    }

                    bw_char_lstm.add_input(word_start);
                    Expression bw_i = bw_char_lstm.back();
                    vector<Expression> tt = {fw_i, bw_i};
                    w = concatenate(tt); //and this goes into the buffer...
                }
            } else { //NO SPELLING
                //Don't use SPELLING
                //cout<<"don't use spelling"<<"\n";
                w = lookup(*hg, p_tok, sent[i]);
            }

            Expression i_i;
            if (USE_POS) {
                Expression p = lookup(*hg, p_pos, pos_seq[i]);
                i_i = affine_transform( { inp_bias, tok2l, w, pos2l, p });
            } else {
                i_i = affine_transform( { inp_bias, tok2l, w });
            }
            if (p_emb && pretrained.count(raw_sent[i])) {
                Expression t = const_lookup(*hg, p_emb, raw_sent[i]);
                i_i = affine_transform( { i_i, emb2l, t });
            }
            // pre-compute buffer representation from left to right
            buffer[sent.size() - i] = rectify(i_i);
            bufferi[sent.size() - i] = i;
        }

        buffer[0] = parameter(*hg, p_buf_guard); // dummy symbol to represent the empty buffer
        bufferi[0] = -999;
        for (auto& b : buffer) {
            buffer_lstm.add_input(b);
        }

        vector < Expression > stack; // variables representing subtree embeddings
        stack.push_back(parameter(*hg, p_stack_guard));
        vector<int> stacki; // position of words in the sentence of head of subtree
        stacki.push_back(-999);
        stack_lstm.add_input(stack.back()); // drive dummy symbol on stack through LSTM

        vector < Expression > sem_stack; // variables representing subtree embeddings
        sem_stack.push_back(parameter(*hg, p_sem_stack_guard));
        vector<int> sem_stacki; // position of words in the sentence of head of subtree
        sem_stacki.push_back(-999);
        sem_stack_lstm.add_input(sem_stack.back()); // drive dummy symbol on stack through LSTM

        vector < Expression > log_probs;
        JointParse partial;
        Parent fake_p = { -1, "ERROR" };
        for (unsigned i = 0; i < raw_sent.size(); i++) {
            partial.syn_arcs[i] = fake_p;
        }

        unsigned act_seq_id = 0;  // incremented at each prediction
        act_name prev_act_enum = MSHIFT; // fake first action so we always start with shift

        while (buffer.size() > 1 || sem_stack.size() > 2) {

            // get list of possible actions for the current parser state
            vector<unsigned> current_valid_actions;

            set < act_name > valid_action_types;
            for (act_name act_enum : corpus.act_types) {
                bool is_forbidden = is_joint_action_forbidden(act_enum,
                        prev_act_enum, buffer.size(), stack.size(),
                        sem_stack.size(), sem_stacki.back(), bufferi.back(),
                        gold_preds, partial);
                if (is_forbidden == false)
                    valid_action_types.insert(act_enum);
            }

            if (valid_action_types.find(PRED) != valid_action_types.end()) {
                assert(valid_action_types.size() == 1);
                unsigned lemma_id = gold_preds.find(bufferi.back())->second;
                auto lpmk = corpus.lemma_practs_map.find(lemma_id);
                if (lpmk == corpus.lemma_practs_map.end()) {
                    current_valid_actions.push_back(
                            corpus.act_dict.Convert(corpus.PR_UNK));
                } else {
                    current_valid_actions = lpmk->second;
                }

            } else {
                for (unsigned a = 0; a < corpus.all_corpus_acts.size(); a++) {
                    const act_name act_type = corpus.all_corpus_acts[a];
                    if (valid_action_types.find(act_type)
                            != valid_action_types.end()) {
                        current_valid_actions.push_back(a);
                    }
                }
            }

            // p_embed = parsact_bias + S * slstm + M * semlstm + B * blstm + A * almst
            Expression parstate = affine_transform( { parsact_bias,
                    synst2next_act, stack_lstm.back(), semst2next_act,
                    sem_stack_lstm.back(), buf2next_act, buffer_lstm.back(),
                    act2next_act, action_lstm.back() });
            Expression rect_parstate = rectify(parstate);

            Expression small_act_bias = select_rows(act_bias,
                    current_valid_actions);
            Expression small_state2next_act = select_rows(state2next_act,
                    current_valid_actions);

            // pars_act = act_bias + parse2next_act * rect_parstate
            Expression pars_act;

            if (build_training_graph && USE_DROPOUT) {
                pars_act = dropout(affine_transform( { small_act_bias,
                        small_state2next_act, rect_parstate }), DROPOUT);
            } else {
                pars_act = affine_transform( { small_act_bias,
                        small_state2next_act, rect_parstate });
            }

            Expression prob_dist = log_softmax(pars_act); //, current_valid_actions);
            vector<float> prob_dist_vec = as_vector(hg->incremental_forward());
            // so it can be iterated over
            // do the argmax
            double best_score = prob_dist_vec[0]; //[current_valid_actions[0]];
            unsigned best_act_id = current_valid_actions[0];
            unsigned best_idx = 0;
            for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
                if (prob_dist_vec[i] > best_score) {
                    // if (prob_dist_vec[current_valid_actions[i]] > best_score) {
                    best_score = prob_dist_vec[i]; //prob_dist_vec[current_valid_actions[i]];
                    best_act_id = current_valid_actions[i];
                    best_idx = i;
                }
            }

            unsigned chosen_act_id = best_act_id;
            unsigned chosen_idx = best_idx;
            if (build_training_graph) { // if we have reference actions (for training) use them
                chosen_act_id = correct_actions[act_seq_id];
                // find the index of the correct action in the list of current valid actions.
                chosen_idx = find(current_valid_actions.begin(),
                        current_valid_actions.end(), chosen_act_id)
                        - current_valid_actions.begin();
                if (chosen_idx >= current_valid_actions.size()) {
                    cerr << "correct action "
                            << corpus.act_dict.Convert(chosen_act_id)
                            << " not in the list of Valid Actions:" << endl;

                    for (unsigned x = 0; x < current_valid_actions.size();
                            x++) {
                        cerr
                                << corpus.act_dict.Convert(
                                        current_valid_actions[x]) << endl;
                    }
                    cerr << "problem in forbidden actions implementation"
                            << endl;
                    exit(1);
                }
                if (best_act_id == chosen_act_id) {
                    (*right)++;
                }
            }

            ++act_seq_id;
            //log_probs.push_back(pick(prob_dist, chosen_act_id));
            log_probs.push_back(pick(prob_dist, chosen_idx));
            results.push_back(chosen_act_id);

            // add current action to action LSTM
            // TODO(Swabha): add functionality to deal with unknown actions
            Expression action_vec = lookup(*hg, p_act, chosen_act_id);
            action_lstm.add_input(action_vec);

            // get relation embedding from action (TODO: convert to syn_label from action?)
            Expression label = lookup(*hg, p_syn_label, chosen_act_id);
            Expression sem_label = lookup(*hg, p_sem_label, chosen_act_id);

            // do action
            const act_name& chosen_act_enum =
                    corpus.all_corpus_acts[chosen_act_id];

            Expression head, mod, composed, nlcomposed;

//			if (!build_training_graph) {
//				cerr << "[";
//				for (unsigned ii = 0; ii < stacki.size(); ++ii) cerr << stacki[ii] << " ";
//				cerr << "][";
//				for (unsigned ii = 0; ii < bufferi.size(); ++ii) cerr << bufferi[ii] << " ";
//				cerr << "]\n[";
//				for (unsigned ii = 0; ii < sem_stacki.size(); ++ii) cerr << sem_stacki[ii] << " ";
//				cerr << "] " << best_action << endl;
//			}

            if (chosen_act_enum == LEFT) { // Syntactic LEFT-ARC
                assert(stack.size() > 1);
                assert(buffer.size() > 1);

                head = buffer.back();
                mod = stack.back();
                composed = affine_transform( { comp_bias, head_comp, head,
                        mod_comp, mod, label_comp, label });
                nlcomposed = tanh(composed);

                Parent par = { bufferi.back(), corpus.act_dict.Convert(
                        chosen_act_id) };
                partial.syn_arcs[stacki.back()] = par;

                stack_lstm.rewind_one_step();
                stack.pop_back();
                stacki.pop_back();

                buffer_lstm.rewind_one_step();
                buffer_lstm.add_input(nlcomposed);
                buffer.pop_back();
                buffer.push_back(nlcomposed);

            } else if (chosen_act_enum == RIGHT) { // Syntactic RIGHT-ARC
                assert(stack.size() > 1);
                assert(buffer.size() > 1);

                head = stack.back();
                mod = buffer.back();
                composed = affine_transform( { comp_bias, head_comp, head,
                        mod_comp, mod, label_comp, label });
                nlcomposed = tanh(composed);

                Parent par = { stacki.back(), corpus.act_dict.Convert(
                        chosen_act_id) };
                partial.syn_arcs[bufferi.back()] = par;

                stack_lstm.rewind_one_step();
                stack_lstm.add_input(nlcomposed);
                stack_lstm.add_input(buffer.back());
                stack.pop_back();
                stack.push_back(nlcomposed);
                stack.push_back(buffer.back());
                stacki.push_back(bufferi.back());

            } else if (chosen_act_enum == REDUCE) { // Syntactic REDUCE
                assert(stack.size() > 1); // dummy symbol means > 1 (not >= 1)
                stack.pop_back();
                stacki.pop_back();
                stack_lstm.rewind_one_step();

            } else if (chosen_act_enum == SHIFT) { // Syntactic SHIFT
                assert(buffer.size() > 1); // dummy symbol means > 2 (not >= 2)
                stack_lstm.add_input(buffer.back());
                stack.push_back(buffer.back());
                stacki.push_back(bufferi.back());

            } else if (chosen_act_enum == SWAP) { // Semantic SWAP
                assert(sem_stack.size() > 2);

                Expression sem_top = sem_stack.back();
                sem_stack.pop_back();
                int sem_topi = sem_stacki.back();
                sem_stacki.pop_back();
                sem_stack_lstm.rewind_one_step();

                Expression sem_next = sem_stack.back();
                sem_stack.pop_back();
                int sem_nexti = sem_stacki.back();
                sem_stacki.pop_back();
                sem_stack_lstm.rewind_one_step();

                sem_stack.push_back(sem_top);
                sem_stack.push_back(sem_next);
                sem_stack_lstm.add_input(sem_top);
                sem_stack_lstm.add_input(sem_next);
                sem_stacki.push_back(sem_topi);
                sem_stacki.push_back(sem_nexti);

            } else if (chosen_act_enum == MSHIFT) { // Semantic SHIFT
                assert(buffer.size() > 1);

                sem_stack_lstm.add_input(buffer.back());
                sem_stack.push_back(buffer.back());
                sem_stacki.push_back(bufferi.back());

                buffer_lstm.rewind_one_step();
                buffer.pop_back();
                bufferi.pop_back();

            } else if (chosen_act_enum == MLEFT) { // Semantic LEFT-ARC
                assert(sem_stack.size() > 1);
                assert(buffer.size() > 1);
                assert(gold_preds.find(bufferi.back()) != gold_preds.end());

                head = buffer.back();
                mod = sem_stack.back();
                composed = affine_transform( { comp2_bias, head_comp2, head,
                        mod_comp2, mod, label_comp2, sem_label });
                nlcomposed = tanh(composed);

                Parent par = { bufferi.back(), corpus.act_dict.Convert(
                        chosen_act_id) };
                vector < Parent > sem_pars;
                if (partial.sem_arcs.find(sem_stacki.back())
                        != partial.sem_arcs.end()) {
                    sem_pars = partial.sem_arcs.find(sem_stacki.back())->second;
                }
                sem_pars.push_back(par);
                partial.sem_arcs[sem_stacki.back()] = sem_pars;

                buffer_lstm.rewind_one_step();
                buffer_lstm.add_input(nlcomposed);
                buffer.pop_back();
                buffer.push_back(nlcomposed);

            } else if (chosen_act_enum == MRIGHT) { // Semantic RIGHT-ARC
                assert(sem_stack.size() > 1);
                assert(buffer.size() > 1);
                assert(gold_preds.find(sem_stacki.back()) != gold_preds.end());

                head = sem_stack.back();
                mod = buffer.back();
                composed = affine_transform( { comp2_bias, head_comp2, head,
                        mod_comp2, mod, label_comp2, sem_label });
                nlcomposed = tanh(composed);

                Parent par = { sem_stacki.back(), corpus.act_dict.Convert(
                        chosen_act_id) };
                vector < Parent > sem_pars;
                if (partial.sem_arcs.find(bufferi.back())
                        != partial.sem_arcs.end()) {
                    sem_pars = partial.sem_arcs.find(bufferi.back())->second;
                }
                sem_pars.push_back(par);
                partial.sem_arcs[bufferi.back()] = sem_pars;

                sem_stack_lstm.rewind_one_step();
                sem_stack_lstm.add_input(nlcomposed);
                // sem_stack_lstm.add_input(mod);

                sem_stack.pop_back();
                sem_stack.push_back(nlcomposed);

            } else if (chosen_act_enum == SELF) { // Semantic SELF-ARC
                assert(buffer.size() > 1);
                assert(gold_preds.find(bufferi.back()) != gold_preds.end());

                head = buffer.back();
                mod = buffer.back();
                composed = affine_transform( { comp2_bias, head_comp2, head,
                        mod_comp2, mod, label_comp2, sem_label });
                nlcomposed = tanh(composed);

                Parent par = { bufferi.back(), corpus.act_dict.Convert(
                        chosen_act_id) };
                vector < Parent > sem_pars;
                if (partial.sem_arcs.find(bufferi.back())
                        != partial.sem_arcs.end()) {
                    sem_pars = partial.sem_arcs.find(bufferi.back())->second;
                }
                sem_pars.push_back(par);
                partial.sem_arcs[bufferi.back()] = sem_pars;

                buffer_lstm.rewind_one_step();
                buffer_lstm.add_input(nlcomposed);
                buffer.pop_back();
                buffer.push_back(nlcomposed);

            } else if (chosen_act_enum == MREDUCE) { // Semantic REDUCE
                assert(sem_stack.size() > 1);
                sem_stack.pop_back();
                sem_stacki.pop_back();
                sem_stack_lstm.rewind_one_step();

            } else if (chosen_act_enum == PRED) {
                assert(buffer.size() > 1);
                assert(gold_preds.find(bufferi.back()) != gold_preds.end());

                head = buffer.back();
                Expression pred = lookup(*hg, p_pred,
                        gold_preds.find(bufferi.back())->second);
                composed = affine_transform( { comp3_bias, head_comp2, head,
                        pred_comp, pred });
                nlcomposed = tanh(composed);

                buffer_lstm.rewind_one_step();
                buffer_lstm.add_input(nlcomposed);
                buffer.pop_back();
                buffer.push_back(nlcomposed);

                partial.pred_pos.insert(bufferi.back());
                partial.predicate_lemmas[bufferi.back()] =
                        corpus.act_dict.Convert(chosen_act_id);

            } else {
                cerr << "Uh-oh! Crazy action "
                        << corpus.act_dict.Convert(chosen_act_id) << endl;
                exit(1);
            }

            prev_act_enum = chosen_act_enum;
        }
        assert(stack.size() == 2); // guard symbol, root
        assert(stacki.size() == 2);

        assert(sem_stack.size() == 2); // guard symbol
        assert(sem_stacki.size() == 2);

        assert(buffer.size() == 1); // guard symbol
        assert(bufferi.size() == 1);

        Expression tot_neglogprob = -sum(log_probs); // last thing added to computation graph
        assert(tot_neglogprob.pg != nullptr);
        return partial;
    }

    /** run beam search // TODO(Miguel): do we need this?
     vector<unsigned> log_prob_parser_beam(ComputationGraph* hg,
     const vector<unsigned>& raw_sent,  // raw sentence
     const vector<unsigned>& sent,  // sent with OOVs replaced
     const vector<unsigned>& sentPos, const vector<string>& setOfActions,
     unsigned beam_size, double* log_prob) {
     abort();
     #if 0
     vector<unsigned> results;
     ParserState init;

     stack_lstm.new_graph(hg);
     buffer_lstm.new_graph(hg);
     action_lstm.new_graph(hg);
     // variables in the computation graph representing the parameters
     Expression pbias = parameter(*hg, p_parsact_bias);
     Expression H = parameter(*hg, p_head_comp);
     Expression D = parameter(*hg, p_mod_comp);
     Expression R = parameter(*hg, p_label_comp);
     Expression cbias = parameter(*hg, p_comp_bias);
     Expression S = parameter(*hg, p_synst2parsact);
     Expression B = parameter(*hg, p_buf2parsact);
     Expression A = parameter(*hg, p_act2parsact);
     Expression ib = parameter(*hg, p_inp_bias);
     Expression w2l = parameter(*hg, p_tok2l);
     Expression p2l;
     if (USE_POS)
     i_p2l = parameter(*hg, p_pos2l);
     Expression t2l;
     if (p_emb2l)
     i_t2l = parameter(*hg, p_emb2l);
     Expression p2a = parameter(*hg, p_parse2next_act);
     Expression abias = parameter(*hg, p_act_bias);
     Expression action_start = parameter(*hg, p_act_start);

     action_lstm.add_input(i_action_start, hg);

     vector<Expression> buffer(sent.size() + 1);// variables representing word embeddings (possibly including POS info)
     vector<int> bufferi(sent.size() + 1);// position of the words in the sentence

     // precompute buffer representation from left to right
     for (unsigned i = 0; i < sent.size(); ++i) {
     assert(sent[i] < VOCAB_SIZE);
     Expression w = lookup(*hg, p_tok, sent[i]);
     Expression i;
     if (USE_POS) {
     Expression p = lookup(*hg, p_pos, sentPos[i]);
     i_i = hg->add_function<AffineTransform>( {i_ib, i_w2l, i_w, i_p2l, i_p});
     } else {
     i_i = hg->add_function<AffineTransform>( {i_ib, i_w2l, i_w});
     }
     if (p_emb && pretrained.count(raw_sent[i])) {
     Expression t = hg->add_const_lookup(p_emb, sent[i]);
     i_i = hg->add_function<AffineTransform>( {i_i, i_t2l, i_t});
     }
     Expression inl = hg->add_function<Rectify>( {i_i});
     buffer[sent.size() - i] = i_inl;
     bufferi[sent.size() - i] = i;
     }
     // dummy symbol to represent the empty buffer
     buffer[0] = parameter(*hg, p_buf_guard);
     bufferi[0] = -999;
     for (auto& b : buffer)
     buffer_lstm.add_input(b, hg);

     vector<Expression> stack;// variables representing subtree embeddings
     vector<int> stacki;// position of words in the sentence of head of subtree
     stack.push_back(parameter(*hg, p_stack_guard));
     stacki.push_back(-999);// not used for anything
     // drive dummy symbol on stack through LSTM
     stack_lstm.add_input(stack.back(), hg);

     init.stack_lstm = stack_lstm;
     init.buffer_lstm = buffer_lstm;
     init.action_lstm = action_lstm;
     init.buffer = buffer;
     init.bufferi = bufferi;
     init.stack = stack;
     init.stacki = stacki;
     init.results = results;
     init.score = 0;
     if (init.stacki.size() ==1 && init.bufferi.size() == 1) {assert(!"bad0");}

     vector<ParserState> pq;
     pq.push_back(init);
     vector<ParserState> completed;
     while (pq.size() > 0) {
     const ParserState cur = pq.back();
     pq.pop_back();
     if (cur.stack.size() == 2 && cur.buffer.size() == 1) {
     completed.push_back(cur);
     if (completed.size() == BEAM_SIZE) break;
     continue;
     }

     // get list of possible actions for the current parser state
     vector<unsigned> current_valid_actions;
     for (auto a: all_act_ids) {
     if (is_action_forbidden(setOfActions[a], cur.buffer.size(), cur.stack.size(), stacki))
     continue;
     current_valid_actions.push_back(a);
     }

     // p_embed = pbias + S * slstm + B * blstm + A * almst
     Expression p_emb = hg->add_function<AffineTransform>( {i_pbias, i_S, cur.stack_lstm.back(), i_B, cur.buffer_lstm.back(), i_A, cur.action_lstm.back()});

     // nlp_t = tanh(p_embed)
     Expression nlp_t = hg->add_function<Rectify>( {i_p_t});

     // r_t = abias + p2a * nlp
     Expression r_t = hg->add_function<AffineTransform>( {i_abias, i_p2a, i_nlp_t});

     //cerr << "CVAs: " << current_valid_actions.size() << " (cur.buf=" << cur.bufferi.size() << " buf.sta=" << cur.stacki.size() << ")\n";
     // adist = log_softmax(r_t)
     hg->add_function<RestrictedLogSoftmax>( {i_r_t}, current_valid_actions);
     vector<float> adist = as_vector(hg->incremental_forward());

     for (auto action : current_valid_actions) {
     pq.resize(pq.size() + 1);
     ParserState& ns = pq.back();
     ns = cur;  // copy current state to new state
     ns.score += adist[action];
     ns.results.push_back(action);

     // add current action to action LSTM
     Expression action = lookup(*hg, p_act, action);
     ns.action_lstm.add_input(i_action, hg);

     // do action
     const string& actionString=setOfActions[action];
     //cerr << "A=" << actionString << " Bsize=" << buffer.size() << " Ssize=" << stack.size() << endl;
     const char ac = actionString[0];
     if (ac =='S') {  // SHIFT
     assert(ns.buffer.size() > 1);// dummy symbol means > 1 (not >= 1)
     ns.stack.push_back(ns.buffer.back());
     ns.stack_lstm.add_input(ns.buffer.back(), hg);
     ns.buffer.pop_back();
     ns.buffer_lstm.rewind_one_step();
     ns.stacki.push_back(cur.bufferi.back());
     ns.bufferi.pop_back();
     } else { // LEFT or RIGHT
     assert(ns.stack.size() > 2);// dummy symbol means > 2 (not >= 2)
     assert(ac == 'L' || ac == 'R');
     Expression dep, head;
     unsigned depi = 0, headi = 0;
     (ac == 'R' ? dep : head) = ns.stack.back();
     (ac == 'R' ? depi : headi) = ns.stacki.back();
     ns.stack.pop_back();
     ns.stacki.pop_back();
     (ac == 'R' ? head : dep) = ns.stack.back();
     (ac == 'R' ? headi : depi) = ns.stacki.back();
     ns.stack.pop_back();
     ns.stacki.pop_back();
     // get relation embedding from action (TODO: convert to relation from action?)
     Expression relation = lookup(*hg, p_syn_label, action);

     // composed = cbias + H * head + D * dep + R * relation
     Expression composed = affine_transform( {cbias, H, head, D, dep, R, relation});
     // nlcomposed = tanh(composed)
     Expression nlcomposed = tanh(composed);
     ns.stack_lstm.rewind_one_step();
     ns.stack_lstm.rewind_one_step();
     ns.stack_lstm.add_input(i_nlcomposed, hg);
     ns.stack.push_back(i_nlcomposed);
     ns.stacki.push_back(headi);
     }
     } // all curent actions
     prune(pq, BEAM_SIZE);
     } // beam search
     assert(completed.size() > 0);
     prune(completed, 1);
     results = completed.back().results;
     assert(completed.back().stack.size() == 2);// guard symbol, root
     assert(completed.back().stacki.size() == 2);
     assert(completed.back().buffer.size() == 1);// guard symbol
     assert(completed.back().bufferi.size() == 1);
     *log_prob = completed.back().score;
     return results;
     #endif
     }**/
};

void signal_callback_handler(int /* signum */) {
    if (requested_stop) {
        cerr << "\nReceived SIGINT again, quitting.\n";
        _exit(1);
    }
    cerr << "\nReceived SIGINT terminating optimization early...\n";
    requested_stop = true;
}

void print_joint_conll(const std::vector<unsigned>& sent,
        const vector<unsigned>& pos_seq, const vector<string>& sent_oov,
        const map<int, string> pred_oov, const JointParse predicted,
        const string out_file_name, bool is_first) {

    ofstream o_str;
    if (is_first == true) {
        o_str.open(out_file_name);
    } else {
        o_str.open(out_file_name, ios::app);
    }

    for (unsigned i = 0; i < sent.size() - 1; ++i) {
//			assert(i < sent_unk.size()
//					&& ((sent[i] == corpus.get_or_add_word(cpyp::Corpus::UNK) && sent_unk[i].size() > 0)
//						|| (sent[i] != corpus.get_or_add_word(cpyp::Corpus::UNK) && sent_unk[i].size() == 0 && int_tok_map.find(sent[i]) != int_tok_map.end())));
        string tok =
                (sent_oov[i].size() > 0) ?
                        sent_oov[i] : corpus.tok_dict.Convert(sent[i]);
        string pos = corpus.pos_dict.Convert(pos_seq[i]);

        assert(predicted.syn_arcs.find(i) != predicted.syn_arcs.end());
        auto head = predicted.syn_arcs.find(i)->second.head + 1;
        if (head == (int) sent.size())
            head = 0;

        auto label = predicted.syn_arcs.find(i)->second.label;
        size_t l1 = label.find('(') + 1;
        size_t l2 = label.rfind(')') - 1;
        label = label.substr(l1, l2 - l1 + 1);

        string fillpred = "_";
        string pred = "_";

        if (predicted.predicate_lemmas.find(i)
                != predicted.predicate_lemmas.end()) {
            fillpred = "Y";
            pred = predicted.predicate_lemmas.find(i)->second;
            size_t lb1 = pred.find('(') + 1;
            size_t lb2 = pred.rfind(')') - 1;
            pred = pred.substr(lb1, lb2 - lb1 + 1);
            if (pred.size() == 3 && pred[0] == 'U' && pred[1] == 'N'
                    && pred[2] == 'K') {
                if (pred_oov.find(i) == pred_oov.end()) {
                    o_str.close();
                    cerr << "Problem with reading oov predicates" << endl;
                    exit(1);
                }
                pred = pred_oov.find(i)->second + ".01";
            }
        }

        o_str << (i + 1) << '\t'       // 1. ID
                << tok << '\t'         // 2. FORM
                << "_" << '\t'         // 3. LEMMA
                << "_" << '\t'         // 4. PLEMMA
                << "_" << '\t'         // 5. POS
                << pos << '\t'         // 6. PPOS
                << "_" << '\t' 	       // 7. FEAT
                << "_" << '\t'         // 8. PFEAT
                << head << '\t'        // 9. HEAD
                << "_" << '\t'        // 10. PHEAD
                << label << '\t'      // 11. DEPREL
                << "_" << '\t'        // 12. PDEPREL
                << fillpred << '\t'   // 13. FILLPRED
                << pred << '\t';      // 14. PRED

        for (auto prd = predicted.pred_pos.begin();
                prd != predicted.pred_pos.end(); ++prd) {
            bool found = false;
            if (predicted.sem_arcs.find(i) != predicted.sem_arcs.end()) {
                vector < Parent > sem_pars = predicted.sem_arcs.find(i)->second;
                for (auto par = sem_pars.begin(); par != sem_pars.end();
                        ++par) {
                    if (par->head == *prd) {
                        found = true;
                        auto lab = par->label;
                        size_t lb1 = lab.find('(') + 1;
                        size_t lb2 = lab.rfind(')') - 1;
                        lab = lab.substr(lb1, lb2 - lb1 + 1);
                        o_str << lab << "\t";
                        break;
                    }
                }
                if (!found) {
                    o_str << "_\t";
                }
            } else {
                o_str << "_\t";
            }
        }
        o_str << endl;
    }
    o_str << endl;
    o_str.close();
}

void run_eval_script(po::variables_map conf, double *synLAS, double *semF1,
        double *macroF1) {
    string gold_conll = conf["gold_conll"].as<string>().c_str();
    string eval_script = conf["eval_script"].as<string>().c_str();

    string err_file = "err.eval";
    string eval_file_name = "joint.eval";
    if (conf.count("out_model")) {
        eval_file_name = conf["out_model"].as<string>() + ".eval";
        err_file = conf["out_model"].as<string>() + ".err.eval";
    }

    string eval_cmd_str = "perl " + eval_script + " -g " + gold_conll + " -s "
            + output_conll + " -q > " + eval_file_name + " 2> " + err_file;

    const char* eval_cmd = eval_cmd_str.c_str();
    int ran_eval = system(eval_cmd);

    if (ran_eval == 0) {
        string las_line = "  Labeled   attachment score:";
        string lab_f1_line = "  Labeled F1:";
        string lab_macrof1_line = "  Labeled macro F1:";
        string f1_str = "";

        std::ifstream eval_file(eval_file_name);
        string line;
        while (getline(eval_file, line)) {
            if (line.compare(0, las_line.length(), las_line) == 0) {
                f1_str = line.substr(line.size() - 7, 5);
                std::string::size_type sz;
                *synLAS = std::stod(f1_str, &sz);
            } else if (line.compare(0, lab_f1_line.length(), lab_f1_line)
                    == 0) {
                f1_str = line.substr(line.size() - 6, line.size());
                std::string::size_type sz;
                *semF1 = std::stod(f1_str, &sz);
            } else if (line.compare(0, lab_macrof1_line.length(),
                    lab_macrof1_line) == 0) {
                f1_str = line.substr(line.size() - 7, 5);
                std::string::size_type sz;
                *macroF1 = std::stod(f1_str, &sz);
            }
        }
        eval_file.close();
    } else {
        cerr << "DID NOT EVALUATE - try make clean" << endl;
    }
}

/**
 * CONLL 2009 style evaluation for precision, recall and f1.
 */
void compute_joint_correct(JointParse gold, JointParse pred,
        double *syn_correct, double *tp, double *fp, double *fn) {
    for (auto itr = gold.syn_arcs.begin(); itr != gold.syn_arcs.end(); ++itr) {
        auto pred_entry = pred.syn_arcs.find(itr->first);
        if (pred_entry == pred.syn_arcs.end())
            continue;
        if (pred_entry->second == itr->second)
            ++(*syn_correct);
    }

    for (auto itr = gold.predicate_lemmas.begin();
            itr != gold.predicate_lemmas.end(); ++itr) {
        bool found = false;
        for (auto itr2 = pred.predicate_lemmas.begin();
                itr2 != pred.predicate_lemmas.end(); ++itr2) {
            if (itr->first == itr2->first && itr->second == itr2->second) {
                *tp += 1.0;
                found = true;
                break;
            }
        }
        if (!found)
            *fn += 1.0;
    }
    for (auto itr = pred.predicate_lemmas.begin();
            itr != pred.predicate_lemmas.end(); ++itr) {
        bool found = false;
        for (auto itr2 = gold.predicate_lemmas.begin();
                itr2 != gold.predicate_lemmas.end(); ++itr2) {
            if (itr->first == itr2->first && itr->second == itr2->second) {
                found = true;
                break;
            }
        }
        if (!found)
            *fp += 1.0;
    }

    for (auto itr = gold.sem_arcs.begin(); itr != gold.sem_arcs.end(); ++itr) {
        auto pred_entry = pred.sem_arcs.find(itr->first);
        if (pred_entry == pred.sem_arcs.end()) {
            *fn += itr->second.size();
            continue;
        }
        auto gold_vec = itr->second;
        auto pred_vec = pred_entry->second;
        *tp += vector_common_count(gold_vec, pred_vec);
        *fp += vector_diff(pred_vec, gold_vec);
        *fn += vector_diff(gold_vec, pred_vec);
    }

    for (auto itr = pred.sem_arcs.begin(); itr != pred.sem_arcs.end(); ++itr) {
        auto gold_entry = gold.sem_arcs.find(itr->first);
        if (gold_entry == gold.sem_arcs.end()) {
            *fp += itr->second.size();
            continue;
        }
    }
}

void do_test_eval(const unsigned beam_size, ParserBuilder parser,
        const unsigned kUNK, set<unsigned> training_vocab,
        po::variables_map conf) {
    auto time_begin = chrono::high_resolution_clock::now();

    double llh = 0;
    double trs = 0; 	  // TODO(Miguel): what's this?
    double right = 0;

    unsigned corpus_size = corpus.num_sents_dev;

    bool is_first = true;
    for (unsigned idx = 0; idx < corpus_size; ++idx) {
        const vector<unsigned>& toks_dev = corpus.tokens_dev[idx];
        const vector<string>& oov_toks_dev = corpus.oov_tokens_dev[idx];
        const vector<unsigned>& pos_dev = corpus.pos_dev[idx];
        const map<int, unsigned>& preds_dev = corpus.preds_dev[idx];
        const map<int, string>& oov_preds_dev = corpus.oov_preds_dev[idx];

        vector<unsigned> tsentence = toks_dev; // TODO(Miguel): what's this?
        if (!USE_SPELLING) {
            for (auto& w : tsentence) {
                if (training_vocab.count(w) == 0) {
                    w = kUNK;
                }
            }
        }

        ComputationGraph cg;
        double log_prob = 0;
        JointParse predicted;
        if (beam_size == 1) {
            predicted = parser.log_prob_parser(&cg, toks_dev, tsentence,
                    pos_dev, preds_dev, vector<unsigned>(), &right);
        } else {
            cerr << "beam size not 1 - unimplemented. Exiting...";
            exit(1);
        }
        llh -= log_prob;
        trs += corpus.correct_act_dev[idx].size();

        print_joint_conll(toks_dev, pos_dev, oov_toks_dev, oov_preds_dev,
                predicted, output_conll, is_first);
        is_first = false;

    }

    double las = 0, semf1 = 0, macrof1 = 0;
    run_eval_script(conf, &las, &semf1, &macrof1);

    auto time_end = chrono::high_resolution_clock::now();
    cerr << "TEST llh=" << llh << " ppl: " << exp(llh / trs) << " err: "
            << (trs - right) / trs << " las: " << las << " sem F1: " << semf1
            << " macro F1: " << macrof1 << "\t[" << corpus_size << " sents in "
            << std::chrono::duration<double, std::milli>(time_end - time_begin).count()
            << " ms]" << endl;
}

void do_training(Model model, ParserBuilder parser,
        set<unsigned> training_vocab, set<unsigned> singletons,
        string param_fname, po::variables_map conf) {

    double best_macrof1 = 0.0;
    bool soft_link_created = false;
    signal(SIGINT, signal_callback_handler);

    SimpleSGDTrainer sgd(&model); // MomentumSGDTrainer sgd(&model);
    sgd.eta_decay = 0.08; // 0.05;

    vector<unsigned> order(corpus.num_sents);
    for (unsigned i = 0; i < corpus.num_sents; ++i) {
        order[i] = i;
    }

    double tot_seen = 0;
    unsigned si = corpus.num_sents;
    unsigned status_every_i_iterations = 100;
    status_every_i_iterations = min(status_every_i_iterations,
            corpus.num_sents);

    unsigned trs = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;

    while (!requested_stop) {
        ++iter;

        for (unsigned tr_idx = 0; tr_idx < status_every_i_iterations;
                ++tr_idx) {
            if (si == corpus.num_sents) {
                si = 0;
                if (first) {
                    first = false;
                } else {
                    sgd.update_epoch();
                }
                cerr << "**SHUFFLE\n";
                random_shuffle(order.begin(), order.end());
            }
            tot_seen += 1;
            const vector<unsigned>& train_sent = corpus.tokens_train[order[si]];
            vector<unsigned> tsentence = train_sent;
            if (UNK_STRATEGY == 1) {
                for (auto& w : tsentence) {
                    if (singletons.count(w) && cnn::rand01() < UNK_PROB) {
                        w = kUNK;
                    }
                }
            }
            const vector<unsigned>& train_pos = corpus.pos_train[order[si]];
            const vector<unsigned>& train_gold_acts =
                    corpus.correct_act_train[order[si]];
            const map<int, unsigned>& train_preds =
                    corpus.preds_train[order[si]];
            ComputationGraph hg;

            parser.log_prob_parser(&hg, train_sent, tsentence, train_pos,
                    train_preds, train_gold_acts, &right);
            double lp = as_scalar(hg.incremental_forward());
            if (lp < 0) {
                cerr << "Log prob < 0 on sentence " << order[si] << ": lp="
                        << lp << endl;
                assert(lp >= 0.0);
            }
            hg.backward();
            sgd.update(1.0);
            llh += lp;
            ++si;
            trs += train_gold_acts.size();
        }
        sgd.status();
        cerr << "update #" << iter << " (epoch "
                << (tot_seen / corpus.num_sents) << ")\t" << " llh: " << llh
                << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs
                << endl;
        llh = trs = right = 0;

        static int logc = 0;
        ++logc;

        if (logc % 25 == 1) { // report on dev set

            unsigned dev_size = corpus.num_sents_dev;
            double llh = 0;
            double trs = 0;
            double right = 0;

            auto t_start = std::chrono::high_resolution_clock::now();

            bool is_first = true;
            for (unsigned idx = 0; idx < dev_size; ++idx) {
                const vector<unsigned>& dev_sent = corpus.tokens_dev[idx];
                const vector<unsigned>& dev_pos = corpus.pos_dev[idx];
                const vector<string>& oov_toks_dev = corpus.oov_tokens_dev[idx];
                const map<int, unsigned>& dev_preds = corpus.preds_dev[idx];
                const map<int, string>& oov_preds_dev =
                        corpus.oov_preds_dev[idx];

                vector<unsigned> dev_sent_unk = dev_sent; // TODO(Miguel): what's this?
                if (!USE_SPELLING) {
                    for (auto& w : dev_sent_unk) {
                        if (training_vocab.count(w) == 0) {
                            w = kUNK;
                        }
                    }
                }

                ComputationGraph hg;
                JointParse predicted = parser.log_prob_parser(&hg, dev_sent,
                        dev_sent_unk, dev_pos, dev_preds, vector<unsigned>(),
                        &right);
                double lp = 0;
                llh -= lp;
                trs += corpus.correct_act_dev[idx].size();

                print_joint_conll(dev_sent, dev_pos, oov_toks_dev,
                        oov_preds_dev, predicted, output_conll, is_first);
                is_first = false;
            }

            double las = 0, semf1 = 0, macrof1 = 0;
            run_eval_script(conf, &las, &semf1, &macrof1);

            auto t_end = std::chrono::high_resolution_clock::now();
            cerr << "  **dev (iter=" << iter << " epoch="
                    << (tot_seen / corpus.num_sents) << ")\t" << " llh=" << llh
                    << " ppl: " << exp(llh / trs) << " err: "
                    << (trs - right) / trs << " las: " << las << " semF1: "
                    << semf1 << " macro:" << macrof1 << "\t[" << dev_size
                    << " sents in "
                    << std::chrono::duration<double, std::milli>(
                            t_end - t_start).count() << " ms]" << endl;

            if (macrof1 > best_macrof1) {
                best_macrof1 = macrof1;
                ofstream out(param_fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
                // Create a soft link to the most recent model in order to make it
                // easier to refer to it in a shell script.
                if (soft_link_created == false) {
                    string softlink = " latest_model";
                    if (conf.count("out_model")) { // if output model file is specified
                        softlink = " " + conf["out_model"].as<string>();
                    }
                    if (system((string("rm -f ") + softlink).c_str()) == 0
                            && system(
                                    (string("ln -s ") + param_fname + softlink).c_str())
                                    == 0) {
                        cerr << "Created " << softlink << " as a soft link to "
                                << param_fname << " for convenience." << endl;
                    }
                    soft_link_created = true;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);

    cerr << "COMMAND:";
    for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i)
        cerr << ' ' << argv[i];
    cerr << endl;

    po::variables_map conf;
    init_command_line(argc, argv, &conf);

    USE_POS = conf.count("use_pos_tags");
    USE_SPELLING = conf.count("use_spelling"); //Miguel

    corpus.USE_SPELLING = USE_SPELLING;
    corpus.USE_LOWERWV = conf.count("use_lowerwv");

    LAYERS = conf["layers"].as<unsigned>();
    INPUT_DIM = conf["input_dim"].as<unsigned>();
    PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
    HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
    ACTION_DIM = conf["action_dim"].as<unsigned>();
    LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
    POS_DIM = conf["pos_dim"].as<unsigned>();
    REL_DIM = conf["rel_dim"].as<unsigned>();
    PRED_DIM = conf["pred_dim"].as<unsigned>();

    BEAM_SIZE = conf["beam_size"].as<unsigned>();

    UNK_STRATEGY = conf["unk_strategy"].as<unsigned>();
    if (UNK_STRATEGY != 1) {
        cerr << "NOT STOCHASTIC REPLACEMENT: unknown word strategy \n"; // TODO: Miguel: What's this?
        abort();
    }

    UNK_PROB = conf["unk_prob"].as<double>();
    assert(UNK_PROB >= 0.);
    assert(UNK_PROB <= 1.);

    if (conf.count("dropout")) {
        USE_DROPOUT = true;
        DROPOUT = conf["dropout"].as<float>();
        cerr << "Using dropout = " << DROPOUT << endl;
    }

    ostringstream os;
    os << "parser_" << (USE_POS ? "pos" : "nopos") << '_' << LAYERS << '_'
            << INPUT_DIM << '_' << HIDDEN_DIM << '_' << ACTION_DIM << '_'
            << LSTM_INPUT_DIM << '_' << POS_DIM << '_' << REL_DIM << "_pid"
            << getpid() << ".params";
    param_fname = os.str();
    cerr << "Parameters will be written to: " << param_fname << endl;
    output_conll = conf["output_conll"].as<string>().c_str();

    // training data is required even at test time to do OOV handling
    corpus.load_correct_actions(conf["training_data"].as<string>());

    VOCAB_SIZE = corpus.tok_dict.size() + 1;
    ACTION_SIZE = corpus.act_dict.size() + 1; // +1 to account for unknown actions
    POS_SIZE = corpus.pos_vocab_size + 10; // To account for extra POS tags we might see in dev. TODO
    PRED_SIZE = corpus.lemma_dict.size() + 1; // +1 to account for unknown predicates
    if (corpus.char_vocab_size > 255) {
        CHAR_SIZE = corpus.char_vocab_size;
    }

    kUNK = corpus.get_or_add_word(cpyp::Corpus::UNK);
    kROOT_SYMBOL = corpus.get_or_add_word(ROOT_SYMBOL);

    // Loading pretrained word embeddings....
    if (conf.count("words")) {
        pretrained[kUNK] = vector<float>(PRETRAINED_DIM, 0);
        cerr << "Loading from " << conf["words"].as<string>() << " with "
                << PRETRAINED_DIM << " dimensions\n";
        ifstream in(conf["words"].as<string>().c_str());

        if (!in.is_open()) {
            cerr << "Pretrained embeddings FILE NOT FOUND!" << endl;
        }
        string line;
        getline(in, line);
        vector<float> v(PRETRAINED_DIM, 0);
        string word;
        while (getline(in, line)) {
            istringstream lin(line);
            lin >> word;

            for (unsigned i = 0; i < PRETRAINED_DIM; ++i)
                lin >> v[i];
            unsigned id = corpus.get_or_add_word(word);
            pretrained[id] = v;
        }
        in.close();
    }
    cerr << "#pretrained embeddings known: " << pretrained.size() << endl;

    // Computing the singletons in the parser's training data for OOV replacement
    set<unsigned> training_vocab; // words available in the training corpus
    set<unsigned> singletons;
    {
        map<unsigned, unsigned> counts;
        for (auto sent : corpus.tokens_train) {
            for (auto word : sent.second) {
                training_vocab.insert(word);
                counts[word]++;
            }
        }
        for (auto wc : counts) {
            if (wc.second == 1) {
                singletons.insert(wc.first);
            }
        }
    }

    Model model;
    ParserBuilder parser(&model, pretrained);
    if (conf.count("model")) {
        ifstream in(conf["model"].as<string>().c_str());
        boost::archive::text_iarchive ia(in);
        ia >> model;
    }

    // OOV words in dev/test data are replaced by UNK
    corpus.load_correct_actions_dev(conf["dev_data"].as<string>());
    if (conf.count("propbank_lemmas")) {
        corpus.load_train_preds(conf["propbank_lemmas"].as<string>());
    }

    // TRAINING
    if (conf.count("train")) {
        cerr << "Training started..." << endl;
        do_training(model, parser, training_vocab, singletons, param_fname,
                conf);
    }

    cerr << "Testing..." << endl;
    do_test_eval(BEAM_SIZE, parser, kUNK, training_vocab, conf);
    return 0;
}
