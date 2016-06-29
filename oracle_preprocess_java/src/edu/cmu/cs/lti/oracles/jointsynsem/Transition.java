package edu.cmu.cs.lti.oracles.jointsynsem;

public class Transition {

    public String label;
    public Action action;

    // default constructor is for the shift action
    public Transition() {
        label = null;// TODO: make optional instead of null. Learn how to use Guava for this.
        action = Action.NONE;
    }

    public Transition(Action action, String label) {
        this.action = action;
        this.label = label;
    }

}
