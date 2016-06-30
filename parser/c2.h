#ifndef CPYPDICT_H_
#define CPYPDICT_H_
#endif

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <vector>
#include <map>

using namespace std;
using namespace badname;
namespace cpyp {
class Corpus {
public:
    bool USE_SPELLING = false;
    bool USE_LOWERWV = false;
    // String literals
    static constexpr const char* UNK = "UNK";
    static constexpr const char* BAD0 = "<BAD0>";

    map<int, vector<unsigned>> correct_act_train;
    map<int, vector<unsigned>> tokens_train;
    map<int, vector<unsigned>> pos_train;
    map<int, map<int, unsigned>> preds_train;
    unsigned num_sents;

    map<int, vector<unsigned>> correct_act_dev;
    map<int, vector<unsigned>> tokens_dev;
    map<int, vector<string>> oov_tokens_dev;
    map<int, vector<unsigned>> pos_dev;
    map<int, map<int, unsigned>> preds_dev;
    map<int, map<int, string>> oov_preds_dev;
    unsigned num_sents_dev;

    Dict act_dict;
    set<act_name> act_types;
    vector<act_name> all_corpus_acts;
    static constexpr const char* PR_UNK = "PR(UNK)";

    unsigned token_vocab_size;
    unsigned pos_vocab_size;
    unsigned char_vocab_size;

    Dict tok_dict;
    Dict pos_dict;
    Dict lemma_dict;
    Dict oov_lemma_dict;

    map<string, unsigned> chars_int_map;
    map<unsigned, string> int_chars_map;

    unordered_map<unsigned, vector<unsigned>> lemma_practs_map;

    Corpus() {
        num_sents = 0;

        num_sents_dev = 0;
        token_vocab_size = 0;
        pos_vocab_size = 0;
        char_vocab_size = 0; // Miguel

    }

    inline unsigned get_UTF8_len(unsigned char x) {
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

    /**
     * 3 kinds of lines : empty between examples, input line and action line
     * the input line is in the format, token1-pos1 token2-pos2~lemma2 ....
     * the words which are predicates are marked as such by ~<lemma>
     */
    inline void load_correct_actions(string file) {
        ifstream actions_file(file);
        string line;

        // checking that everything is empty when train data is read...
        assert(token_vocab_size == 0);
        assert(pos_vocab_size == 0);
        assert(lemma_dict.size() == 0);
        unsigned num_tokens = 0;

        chars_int_map[BAD0] = 1;
        int_chars_map[1] = "BAD0";
        char_vocab_size = 1;

        vector<unsigned> current_sent_tok;
        vector<unsigned> current_sent_pos;
        map<int, unsigned> current_sent_lemmas;

        int sent_idx = -1;
        bool input_line = false;
        bool first_ex = true;

        string token, pos, pred, tok_pos_pair;

        while (getline(actions_file, line)) {
            replace_str_in_place(line, "-RRB-", "_RRB_");
            replace_str_in_place(line, "-LRB-", "_LRB_");

            if (line.empty()) { // end of example
                if (!first_ex) {
                    tokens_train[sent_idx] = current_sent_tok;
                    pos_train[sent_idx] = current_sent_pos;
                    preds_train[sent_idx] = current_sent_lemmas;
                } else {
                    first_ex = false;
                }
                sent_idx++;
                input_line = true;
                current_sent_tok.clear();
                current_sent_pos.clear();
                current_sent_lemmas.clear();
            } else if (input_line) {
                // read inp line tok by tok: "the-det," "cat-noun, "stinks-verb~stink"...

                istringstream iss(line);
                if (sent_idx % 1000 == 0)
                    cerr << sent_idx << "...";

                do {
                    tok_pos_pair.clear();
                    iss >> tok_pos_pair;
                    if (tok_pos_pair.size() == 0) {
                        continue;
                    }
                    // remove the trailing comma if need be.
                    if (tok_pos_pair[tok_pos_pair.size() - 1] == ',') {
                        tok_pos_pair = tok_pos_pair.substr(0,
                                tok_pos_pair.size() - 1);
                    }

                    // split the string (at '-') into word and POS tag( + predicate if true)
                    size_t postag_char_idx = tok_pos_pair.rfind('-'); //TODO(Miguel): what if the word itself is "-" ??
                    size_t pred_char_idx = tok_pos_pair.rfind('~');

                    // POS tag
                    assert(postag_char_idx != string::npos);
                    pos = tok_pos_pair.substr(postag_char_idx + 1,
                            pred_char_idx - 1 - postag_char_idx);
                    unsigned pos_id = pos_dict.Convert(pos);
                    pos_vocab_size = pos_dict.size();
                    current_sent_pos.push_back(pos_id);

                    // Token
                    token = tok_pos_pair.substr(0, postag_char_idx);
                    if (USE_LOWERWV) {
                        transform(token.begin(), token.end(), token.begin(), ::tolower);
                    }
                    if (!tok_dict.Contains(token)) {
                        // character stuff // TODO(Swabha): look into later
                        unsigned j = 0;
                        while (j < token.length()) {
                            string jth_char = "";
                            for (unsigned h = j; h < j + get_UTF8_len(token[j]);
                                    h++) {
                                jth_char += token[h];
                            }
                            if (chars_int_map[jth_char] == 0) {
                                chars_int_map[jth_char] = char_vocab_size;
                                int_chars_map[char_vocab_size] = jth_char;
                                char_vocab_size++;
                            }
                            j += get_UTF8_len(token[j]);
                        }
                    }
                    unsigned tok_id = tok_dict.Convert(token);
                    token_vocab_size = tok_dict.size();
                    num_tokens++;
                    current_sent_tok.push_back(tok_id);

                    // Predicate
                    if (pred_char_idx != string::npos) {
                        pred = tok_pos_pair.substr(pred_char_idx + 1);
                        lemma_dict.Convert(pred);
                        current_sent_lemmas[current_sent_tok.size() - 1] =
                                lemma_dict.Convert(pred);
                    }
                } while (iss);
                input_line = false;
            } else if (!input_line) { // actions
                unsigned act_idx = act_dict.Convert(line);
                vector<unsigned> a = correct_act_train[sent_idx];
                a.push_back(act_idx);
                correct_act_train[sent_idx] = a;
            }
        }
        // Add the last sentence.
        if (current_sent_tok.size() > 0) {
            tokens_train[sent_idx] = current_sent_tok;
            pos_train[sent_idx] = current_sent_pos;
            preds_train[sent_idx] = current_sent_lemmas;
            sent_idx++;

        }
        actions_file.close();
        num_sents = sent_idx;

        cerr << "done reading training actions file\n";
        cerr << "#sents: " << num_sents << endl;
        cerr << "#tokens: " << num_tokens << endl;
        cerr << "#types: " << token_vocab_size << endl;
        cerr << "#POStags: " << pos_vocab_size << endl;
        cerr << "#actions: " << act_dict.size() << endl;
        cerr << "#preds:" << lemma_dict.size() << endl;
    }

    inline unsigned get_or_add_word(const string& word) {
        return tok_dict.Convert(word);
    }

    /** Hack to deal with unseen actions labels - replace the label with some other label */
    void get_some_act_label(map<act_name, string>& act_labels_map) {
        for (unsigned idx = 0; idx < all_corpus_acts.size(); ++idx) {
            act_name aname = all_corpus_acts[idx];
            auto act_id = act_labels_map.find(aname);
            if (act_id == act_labels_map.end()) {
                string act_string = act_dict.Convert(idx);
                act_labels_map[aname] = act_string;
            }
        }
    }

    inline void load_correct_actions_dev(string file) {
        tok_dict.Freeze(); // contains all tokens in train set, and if used, pre-trained embedding
        tok_dict.SetUnk(UNK);

        act_dict.Freeze();
        act_dict.SetUnk(PR_UNK);
        get_all_acts(act_dict, all_corpus_acts, act_types);

        map < act_name, string > some_label;
        get_some_act_label (some_label);

        lemma_dict.Freeze();
        lemma_dict.SetUnk(UNK);

        ifstream dev_actions_file(file);
        string dev_line;
        assert(pos_vocab_size > 1);
        assert(token_vocab_size > 3);

        vector<unsigned> current_sent;
        vector < string > current_sent_oov;
        vector<unsigned> current_posseq;
        map<int, unsigned> current_predmap;
        map<int, string> current_predmap_oov;

        int sent_idx = -1;
        bool inp_line = false;
        bool first_ex = true;

        string token, pos, pred, tok_pos_pair;
        unsigned num_tokens = 0;

        while (getline(dev_actions_file, dev_line)) {
            replace_str_in_place(dev_line, "-RRB-", "_RRB_");
            replace_str_in_place(dev_line, "-LRB-", "_LRB_");

            if (dev_line.empty()) { // end of example
                if (!first_ex) {
                    tokens_dev[sent_idx] = current_sent;
                    oov_tokens_dev[sent_idx] = current_sent_oov;
                    pos_dev[sent_idx] = current_posseq;
                    preds_dev[sent_idx] = current_predmap;
                    oov_preds_dev[sent_idx] = current_predmap_oov;
                } else {
                    first_ex = false;
                }
                sent_idx++;
                inp_line = true;
                current_sent.clear();
                current_sent_oov.clear();
                current_posseq.clear();
                current_predmap.clear();
                current_predmap_oov.clear();
            } else if (inp_line) {
                // read inp line tok by tok

                istringstream iss(dev_line);
                if (sent_idx % 1000 == 0)
                    cerr << sent_idx << "...";

                do {
                    tok_pos_pair.clear();
                    iss >> tok_pos_pair;
                    if (tok_pos_pair.size() == 0) {
                        continue;
                    }
                    // remove the trailing comma if need be.
                    if (tok_pos_pair[tok_pos_pair.size() - 1] == ',') {
                        tok_pos_pair = tok_pos_pair.substr(0,
                                tok_pos_pair.size() - 1);
                    }
                    // split the string (at '-') into word and POS tag.
                    size_t postag_charpos = tok_pos_pair.rfind('-');
                    size_t pred_charpos = tok_pos_pair.rfind('~');

                    // POS tag
                    assert(postag_charpos != string::npos);
                    pos = tok_pos_pair.substr(postag_charpos + 1,
                            pred_charpos - 1 - postag_charpos);
                    unsigned pos_id = pos_dict.Convert(pos);
                    pos_vocab_size = pos_dict.size();
                    current_posseq.push_back(pos_id);

                    // Token
                    token = tok_pos_pair.substr(0, postag_charpos);
                    if (USE_LOWERWV) {
                        transform(token.begin(), token.end(), token.begin(), ::tolower);
                    }
                    unsigned tok_id = tok_dict.Convert(token);
                    token_vocab_size = tok_dict.size();
                    ++num_tokens;
                    current_sent.push_back(tok_id);

                    // OOV Token
                    // add an empty string for any token except OOVs (it is easy to
                    // recover the surface form of non-OOV using int_words_map(id)).
                    current_sent_oov.push_back("");
                    if (tok_dict.is_unk(tok_id)) {
                        if (USE_SPELLING) {
                            cerr << "UNK word dict not implemented" << endl;
                            cerr << "Spelling model will fail";
                            exit(1);
                        } else {
                            // save the surface form of this OOV as a string
                            current_sent_oov[current_sent_oov.size() - 1] =
                                    token;
                        }
                    }

                    // Predicate
                    if (pred_charpos != string::npos) {
                        pred = tok_pos_pair.substr(pred_charpos + 1);
                        unsigned pred_id = lemma_dict.Convert(pred);
                        current_predmap[current_sent.size() - 1] = pred_id;
                        if (lemma_dict.is_unk(pred_id)) {
                            oov_lemma_dict.Convert(pred);
                            current_predmap_oov[current_sent.size() - 1] = pred;
                        }
                    }
                } while (iss);
                inp_line = false;
            } else if (inp_line == false) {
                if (!act_dict.Contains(dev_line)) {
                    if (dev_line[1] == 'R' && dev_line[3] == 'A'
                            && dev_line[4] == 'A') {
                        // TODO(Swabha): Change - Nasty hack for replacing SR(AA) with SR(A1)
                        dev_line = "SR(A1)";
                    } else if (dev_line[0] == 'P' && dev_line[1] == 'R'
                            && dev_line[2] == '(') {
                        dev_line = UNK;
                    } else {
                        size_t sepIndex = dev_line.rfind('|');
                        if (sepIndex == string::npos
                                && !act_dict.Contains(dev_line)) {
                            cerr << "Unknown action " << dev_line;
                            act_name aname = get_act_name(dev_line);
                            dev_line = some_label[aname];
                            cerr << " replaced with " << dev_line << endl;
                            //exit(1);
                        } else {
                            // These labels exist in the data because of projectivization
                            // split the action into 2 parts - e.g. RA(X|Y) into RA(X) and RA(Y)
                            cerr << dev_line << " fixable action" << endl;
                            string a_str_left = dev_line.substr(0, sepIndex)
                                    + ")";
                            if (act_dict.Contains(a_str_left)) {
                                dev_line = a_str_left;
                            } else {
                                size_t sepIndex2 = dev_line.rfind('(');
                                string a_str_right = dev_line.substr(0,
                                        sepIndex2)
                                        + dev_line.substr(sepIndex + 1,
                                                dev_line.npos);
                                if (act_dict.Contains(a_str_right)) {
                                    dev_line = a_str_right;
                                } else {
                                    cerr << "What on earth is this? "
                                            << dev_line << endl;
                                    exit(1);
                                }
                            }
                        }
                    }
                }

                unsigned act_idx = act_dict.Convert(dev_line);
                vector<unsigned> a = correct_act_dev[sent_idx];
                a.push_back(act_idx);
                correct_act_dev[sent_idx] = a;

            }
        }
        // Add the last sentence.
        if (current_sent.size() > 0) {
            tokens_dev[sent_idx] = current_sent;
            oov_tokens_dev[sent_idx] = current_sent_oov;
            pos_dev[sent_idx] = current_posseq;
            preds_dev[sent_idx] = current_predmap;
            oov_preds_dev[sent_idx] = current_predmap_oov;
            sent_idx++;
        }
        dev_actions_file.close();
        cerr << "done reading dev/test actions file\n";

        num_sents_dev = sent_idx;
        cerr << "#sents: " << num_sents_dev << endl;
        cerr << "#tokens: " << num_tokens << endl;
        cerr << "#types: " << token_vocab_size << endl;
        cerr << "#POStags: " << pos_vocab_size << endl;
        cerr << "#actions: " << act_dict.size() << endl;
        cerr << "#action types: " << act_types.size() << endl;
        cerr << "#preds:" << lemma_dict.size() << endl;
        cerr << "#OOV preds:" << oov_lemma_dict.size() << endl;
    }

    void replace_str_in_place(string& subject, const string& search,
            const string& replace) {
        size_t pos = 0;
        while ((pos = subject.find(search, pos)) != string::npos) {
            subject.replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    inline void load_train_preds(string file) {
        ifstream train_preds_file(file);
        string line;
        string inp_str;

        assert(lemma_dict.is_frozen());
        assert(act_dict.is_frozen());

        vector<unsigned> pred_acts;
        while (getline(train_preds_file, line)) {
            istringstream iss(line);
            bool isfirst = true;
            unsigned lemma_id;
            pred_acts.clear();
            do {
                iss >> inp_str;
                if (isfirst) {
                    if (lemma_dict.Contains(inp_str)) {
                        lemma_id = lemma_dict.Convert(inp_str);
                    } else {
                        cerr << "lemma not found " << inp_str << endl;
                    }
                    isfirst = false;
                } else {
                    if (act_dict.Contains(inp_str)) {
                        pred_acts.push_back(act_dict.Convert(inp_str));
                    } else {
                        cerr << "predicate act " << inp_str << " not found"
                                << endl;
                    }
                }
            } while (iss);
            pred_acts.pop_back();
            lemma_practs_map[lemma_id] = pred_acts;
        }
        train_preds_file.close();
        cerr << "#lemmas with associated PR acts = " << lemma_practs_map.size()
                << endl;
    }
};
}
