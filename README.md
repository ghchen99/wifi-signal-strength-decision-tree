# WiFi Signal Strength Decision Tree
This project implements a decision tree algorithm to determine indoor locations based on WiFi signal strengths collected from a mobile phone. The goal is to create a decision tree that can handle continuous attributes and multiple labels. The implementation includes loading the dataset, creating decision trees, evaluating the tree's performance, and optional pruning.

<p align="center">
<img width="236" alt="Screenshot 2023-06-01 at 15 30 42" src="https://github.com/ghchen99/wifi-signal-strength-decision-tree/assets/56446026/b70b15ac-4452-44d6-bbae-779af0e78536">
 </p>

Figure 1: Illustration of the scenario. The WIFI signal strength from 7 emitters are recorded from a mobile phone. The objective of this coursework is to learn decision tree that predict in which of the 4 rooms the user is standing.

## Loading Data
The dataset can be loaded from the files "WIFI db/clean dataset.txt" and "WIFI db/noisy dataset.txt". Each file contains a 2000x8 array representing a dataset of 2000 samples. Each sample consists of 7 WiFi signal strengths, and the last column indicates the room number (label) of the sample. The dataset contains continuous attributes, and the text files can be loaded using the loadtxt function from NumPy.

## Creating Decision Trees
To create the decision tree, a recursive function called decision_tree_learning() is implemented. It takes a dataset matrix and a depth variable as arguments. The function follows the pseudo-code described in "Artificial Intelligence: A Modern Approach" by Stuart Russell and Peter Norvig, with modifications to handle continuous attributes.

The decision_tree_learning() function follows the algorithm described below:

```sql
procedure decision_tree_learning(training_dataset, depth)
    if all samples have the same label then
        return (a leaf node with this value, depth)
    else
        split ← find_split(training_dataset)
        node ← a new decision tree with root as split value
        l_branch, l_depth ← decision_tree_learning(l_dataset, depth+1)
        r_branch, r_depth ← decision_tree_learning(r_dataset, depth+1)
        return (node, max(l_depth, r_depth))
    end if
end procedure
'''

The find_split function chooses the attribute and value that results in the highest information gain. Since the dataset contains continuous attributes, the algorithm searches for the split point that provides the highest information gain. To find good split points efficiently, the attribute values are sorted, and split points are considered between two examples in sorted order while keeping track of positive and negative examples on each side of the split point.

Information gain is evaluated by computing the label distribution (or probability) for each subset of the dataset based on the splitting rule. The information gain is defined using the general definition of entropy. The entropy and remainder functions are used to calculate the information gain as follows:


