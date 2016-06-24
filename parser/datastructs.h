#include <iostream>
#include <string>
#include <cassert>

#include <boost/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#if BOOST_VERSION >= 105600
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unordered_map.hpp>
#endif

using namespace std;

namespace badname {

template<typename Map>
static bool map_compare(Map const &lhs, Map const &rhs) {
    return lhs.size() == rhs.size()
            && equal(lhs.begin(), lhs.end(), rhs.begin());
}

struct Parent {
    int head;
    string label;

    bool operator ==(const Parent& other) const {
        return (head == other.head) && (label.compare(other.label) == 0);
    }

    bool operator <(const Parent& other) const {
        return head < other.head || (label < other.label);
    }
};

static bool vector_compare(vector<Parent> lhs, vector<Parent> rhs) {
    if (lhs.size() != rhs.size())
        return false;
    sort(lhs.begin(), lhs.end());
    sort(rhs.begin(), rhs.end());
    return lhs == rhs;
}

static unsigned vector_common_count(vector<Parent>& lhs, vector<Parent>& rhs) {
    if (lhs.size() == 0 || rhs.size() == 0) {
        return 0;
    }
    sort(lhs.begin(), lhs.end());
    sort(rhs.begin(), rhs.end());
    vector<Parent> result;
    set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
            back_inserter(result));
    return result.size();
}

/** returns length of lhs -rhs */
static unsigned vector_diff(vector<Parent>& lhs, vector<Parent>& rhs) {
    if (lhs.size() == 0 || rhs.size() == 0) {
        return 0;
    }
    sort(lhs.begin(), lhs.end());
    sort(rhs.begin(), rhs.end());
    vector<Parent> result;
    set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
            back_inserter(result));
    return result.size();
}

struct JointParse {
    map<int, Parent> syn_arcs;
    map<int, vector<Parent>> sem_arcs;
    map<int, string> predicate_lemmas; // This should be sorted, temp fix by also having the vector below
    set<int> pred_pos;

    bool contains_syn_arc(int child, int parent, string label) {
        if (syn_arcs.find(child) == syn_arcs.end())
            return false;
        Parent existing = syn_arcs.find(child)->second;
        Parent test { parent, label };
        if (existing == test)
            return true;
        return false;
    }

    bool contains_sem_arc(int child, int parent) {
        if (sem_arcs.find(child) == sem_arcs.end())
            return false;
        vector<Parent> existing = sem_arcs.find(child)->second;
        for (auto par : existing) {
            if (par.head == parent)
                return true;
            // Note we don't check for the label
        }
        return false;
    }

    bool operator ==(const JointParse& other) const {
        // semantic comparison
        if (sem_arcs.size() != other.sem_arcs.size()) {
            cerr << "sem map sizes are different" << endl;
            return false;
        }
        for (auto itr = sem_arcs.begin(); itr != sem_arcs.end(); ++itr) {
            auto other_entry = other.sem_arcs.find(itr->first);
            if (other_entry == other.sem_arcs.end()) {
                cerr << "arg not found in other parse" << endl;
                return false;
            }
            if (vector_compare(itr->second, other_entry->second) == false) {
                cerr << "semantic parents of argument differ" << endl;
                return false;
            }
        }

        // syntactic comparison
        return map_compare(syn_arcs, other.syn_arcs);
    }

    void print() {
        cerr << "predicates:" << endl;
        for (auto itr = predicate_lemmas.begin(); itr != predicate_lemmas.end();
                ++itr) {
            cerr << itr->first << ":" << itr->second << endl;
        }

        cerr << "sem-arcs:" << endl;
        for (auto itr = sem_arcs.begin(); itr != sem_arcs.end(); ++itr) {
            cerr << itr->first << "<- ";
            for (auto par : itr->second) {
                cerr << par.head << "[" << par.label << "], ";
            }
            cerr << endl;
        }

        cerr << "syn-arcs:" << endl;
        for (auto itr = syn_arcs.begin(); itr != syn_arcs.end(); ++itr) {
            cerr << itr->second.head << "-" << itr->second.label << "->"
                    << itr->first << endl;
        }

    }
};

class Dict {
    typedef unordered_map<string, unsigned> Map;
public:
    Dict() :
            frozen(false), map_unk(false), unk_id(-1) {
    }

    inline unsigned size() const {
        return words_.size();
    }

    void printKeys() {
        for (auto idx = d_.begin(); idx != d_.end(); ++idx) {
            cerr << idx->first << endl;
        }
    }

    inline bool Contains(const string& words) {
        return !(d_.find(words) == d_.end());
    }

    void Freeze() {
        frozen = true;
    }
    bool is_frozen() {
        return frozen;
    }

    bool is_unk(const unsigned wordid) {
        if (wordid == unk_id) {
            return true;
        }
        return false;
    }

    inline unsigned Convert(const string& word) {
        auto i = d_.find(word);
        if (i == d_.end()) {
            if (frozen) {
                if (map_unk) {
                    return unk_id;
                } else {
                    cerr << map_unk << endl;
                    cerr << "Unknown word encountered: " << word << endl;
                    throw runtime_error(
                            "Unknown word encountered in frozen dictionary: "
                                    + word);
                }
            }
            words_.push_back(word);
            return d_[word] = words_.size() - 1;
        } else {
            return i->second;
        }
    }

    inline const string& Convert(const unsigned& id) const {
        assert(id < (unsigned int) words_.size());
        return words_[id];
    }

    void SetUnk(const string& word) {
        if (!frozen)
            throw runtime_error(
                    "Please call SetUnk() only after dictionary is frozen");
        if (map_unk)
            throw runtime_error("Set UNK more than one time");

        // temporarily unfrozen the dictionary to allow the add of the UNK
        frozen = false;
        unk_id = Convert(word);
        frozen = true;

        map_unk = true;
    }

    void addFreq(const string& word) {
        if (Convert(word) == unk_id)
            return;
        if (d_.find(word) == d_.end()) {
            throw runtime_error("Woah something wrong -> " + word);
        }
        int frequency = 1;
        if (freqs.find(d_[word]) != freqs.end()) {
            frequency = freqs[d_[word]] + 1;
        }
        freqs[d_[word]] = frequency;
    }

    bool isSingleton(const unsigned wordid) {
        if (freqs.find(wordid) == freqs.end()) {
            throw runtime_error("Unknown word with 0 freq: " + Convert(wordid));
        }
        if (freqs[wordid] == 1)
            return true;
        return false;
    }

    void clear() {
        words_.clear();
        d_.clear();
    }

private:
    bool frozen;
    bool map_unk; // if true, map unknown word to unk_id
    unsigned unk_id;
    vector<string> words_;
    Map d_;
    unordered_map<unsigned, int> freqs;

    friend class boost::serialization::access;
#if BOOST_VERSION >= 105600
    template<class Archive> void serialize(Archive& ar, const unsigned int) {
        ar & frozen;
        ar & map_unk;
        ar & unk_id;
        ar & words_;
        ar & d_;
    }
#else
    template<class Archive> void serialize(Archive& ar, const unsigned int) {
        throw invalid_argument(
                "Serializing dictionaries is only supported on versions of boost 1.56 or higher");
    }
#endif
};

}
