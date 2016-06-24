#include <iostream>
#include <string>
#include "datastructs.h"

enum act_name {
    SHIFT, REDUCE, LEFT, RIGHT, SWAP, MSHIFT, MLEFT, MRIGHT, MREDUCE, SELF, PRED
};

static act_name get_act_name(const std::string& act_str) {
    const char ac = act_str[0];
    const char ac2 = act_str[1];
    if (ac == 'L')
        return LEFT;
    if (ac == 'M')
        return MREDUCE;
    if (ac == 'P')
        return PRED;
    if (ac == 'R' && ac2 == 'A')
        return RIGHT;
    if (ac == 'R' && ac2 == 'E')
        return REDUCE;
    assert(ac == 'S');
    if (ac == 'S' && ac2 == 'H')
        return SHIFT;
    if (ac == 'S' && ac2 == 'L')
        return MLEFT;
    if (ac == 'S' && ac2 == 'R')
        return MRIGHT;
    if (ac == 'S' && ac2 == 'S')
        return MSHIFT;
    if (ac == 'S' && ac2 == 'W')
        return SWAP;
    if (ac == 'S' && ac2 == 'E')
        return SELF;
    else {
        std::cerr << "Invalid action!!!" << std::endl;
        exit(1);
    }
}

void get_all_acts(badname::Dict act_dict, vector<act_name>& all_corpus_acts,
        set<act_name>& act_types) {
    if (act_dict.is_frozen() == false) {
        cerr << "cannot do this unless frozen" << endl;
        exit(1);
    }
    for (unsigned a = 0; a < act_dict.size(); ++a) {
        act_name act_enum = get_act_name(act_dict.Convert(a));
        all_corpus_acts.push_back(act_enum);
        act_types.insert(act_enum);
    }
}

bool is_syn_action(const act_name action) {
    return (action == SHIFT || action == REDUCE || action == LEFT
            || action == RIGHT);
}

bool go_to_syn_action(const act_name prev_action) {
    return (prev_action == MSHIFT
            || (prev_action != SHIFT && prev_action != RIGHT
                    && is_syn_action(prev_action)));
}

bool go_to_sem_action(const act_name prev_action) {
    return (prev_action == SHIFT || prev_action == RIGHT
            || (prev_action != MSHIFT && !is_syn_action(prev_action)));
}

bool is_joint_action_forbidden(const act_name action,
        const act_name prev_action, unsigned buf_size, unsigned stack_size,
        unsigned sem_stack_size, int sem_stack_top, int buffer_top,
        std::map<int, unsigned> gold_preds, badname::JointParse partial) {

    if (action == LEFT && (stack_size <= 1 || buf_size <= 1))
        return true;
    if (action == RIGHT && (stack_size <= 1 || buf_size <= 2)) // |buf| = 2 means root is modifier
        return true;
    if (action == REDUCE && stack_size <= 1)
        return true;
    if (action == SHIFT && buf_size <= 1)
        return true;
    if (action == SWAP && sem_stack_size <= 2)
        return true;
    if (action == MLEFT && (sem_stack_size <= 1 || buf_size <= 2)) // when |buf| = 2, ROOT is head
        return true;
    if (action == MRIGHT && (sem_stack_size <= 1 || buf_size <= 2)) // when |buf| = 2, ROOT is mod
        return true;
    if (action == MREDUCE && sem_stack_size <= 1)
        return true;
    if (action == MSHIFT && buf_size == 1)
        return true;
    if (action == SELF && buf_size == 1)
        return true;
    if (action == PRED && buf_size == 1)
        return true;

    // previous action should not be the same as these actions - leads to infinite loop
    if (action == SWAP && prev_action == SWAP)
        return true;

    // adding arcs/predicates when they are already added
    if (action == MLEFT && partial.contains_sem_arc(sem_stack_top, buffer_top))
        return true;
    if (action == MRIGHT && partial.contains_sem_arc(buffer_top, sem_stack_top))
        return true;
    if (action == SELF && partial.contains_sem_arc(buffer_top, buffer_top))
        return true;
    if (action == PRED
            && partial.pred_pos.find(buffer_top) != partial.pred_pos.end())
        return true;

    // conditions involving gold pred positions
    if (action == MRIGHT && gold_preds.find(sem_stack_top) == gold_preds.end())
        return true;
    if ((action == MLEFT || action == SELF || action == PRED)
            && gold_preds.find(buffer_top) == gold_preds.end())
        return true;
    if (gold_preds.find(buffer_top) != gold_preds.end()
            && partial.pred_pos.find(buffer_top) == partial.pred_pos.end()
            && (prev_action == SHIFT || prev_action == RIGHT) && action != PRED)
        return true;

    // conditions involving edge cases
    // ROOT is the only thing remaining on buffer, but there are many things on the stack
    if (action == SHIFT && buf_size == 2 && stack_size > 1)
        return true;
    if (action == MSHIFT && buf_size == 2 && sem_stack_size > 1)
        return true;

    // conditions for syntactic-semantic transfer
    if (prev_action == SHIFT && action == RIGHT)
        return true;
    if (is_syn_action(action) && !go_to_syn_action(prev_action))
        return true;
    if (!is_syn_action(action) && !go_to_sem_action(prev_action))
        return true;

    // all ok
    return false;
}
