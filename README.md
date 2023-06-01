# WiFi Signal Strength Decision Tree
This project implements a decision tree algorithm to determine indoor locations based on WiFi signal strengths collected from a mobile phone. The goal is to create a decision tree that can handle continuous attributes and multiple labels. The implementation includes loading the dataset, creating decision trees, evaluating the tree's performance, and optional pruning.

<p align="center">
<img width="236" alt="Screenshot 2023-06-01 at 15 30 42" src="https://github.com/ghchen99/wifi-signal-strength-decision-tree/assets/56446026/b70b15ac-4452-44d6-bbae-779af0e78536">
 </p>

Figure 1: Illustration of the scenario. The WIFI signal strength from 7 emitters are recorded from a mobile phone. The objective of this coursework is to learn decision tree that predict in which of the 4 rooms the user is standing.

## Loading Data
The dataset can be loaded from the files `WIFI db/clean dataset.txt` and `WIFI db/noisy dataset.txt`. Each file contains a 2000x8 array representing a dataset of 2000 samples. Each sample consists of 7 WiFi signal strengths, and the last column indicates the room number (label) of the sample. The dataset contains continuous attributes, and the text files can be loaded using the loadtxt function from NumPy.

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
```

The find_split function chooses the attribute and value that results in the highest information gain. Since the dataset contains continuous attributes, the algorithm searches for the split point that provides the highest information gain. To find good split points efficiently, the attribute values are sorted, and split points are considered between two examples in sorted order while keeping track of positive and negative examples on each side of the split point.

Information gain is evaluated by computing the label distribution (or probability) for each subset of the dataset based on the splitting rule. The information gain is defined using the general definition of entropy. The entropy and remainder functions are used to calculate the information gain as follows:

```scss
Gain(S_all, S_left, S_right) = H(S_all) - Remainder(S_left, S_right)

H(dataset) = - ∑(pk * log2(pk))

Remainder(S_left, S_right) = |S_left| * H(S_left) + |S_right| * H(S_right) / (|S_left| + |S_right|)
```
The implementation of the decision tree (in Python) uses dictionaries to store nodes as a single object. Each node has attributes like "attribute", "value", "left", and "right". The decision tree can be visualized using the matplotlib library.

## Evaluation
The performance of the decision tree is evaluated using 10-fold cross-validation on both the clean and noisy datasets. The evaluation includes:

- Confusion matrix: A 4x4 matrix representing the classification results.
- Average recall and precision rates per class: Derived from the confusion matrix.
- F1-measures: Derived from the recall and precision rates.
- Average classification rate: Calculated as 1 - classification error.

The results for both datasets are reported, including the accuracy of room recognition, confusion between rooms, and any notable observations.

## Pruning
To reduce the performance difference between the clean and noisy datasets, a pruning function based on reducing the validation error is implemented. The pruning process involves substituting nodes directly connected to two leaves with a single leaf if it reduces the validation error. The tree is iteratively pruned until no nodes connected to two leaves remain.

The performances of the decision tree before and after pruning are compared using 10-fold cross-validation. The results are reported, and any observations or insights about the effect of pruning are discussed.

## Implementation Details
The implementation of the decision tree and evaluation functions only used the numpy and matplotlib libraries. Other libraries like scikit-learn were not allowed. The code consists of four main functions:

1. `decision_tree_learning(dataset, depth)`: This function creates a decision tree by recursively splitting the dataset based on the information gain. The function takes a dataset and depth as input and returns the root node of the tree along with the maximum depth.

2. `evaluate(dataset, tree)`: This function performs the evaluation of the decision tree using a nested 10-fold cross-validation. It also includes the pruning process. Please note that running this function will take approximately 45 minutes to complete due to the creation, pruning, and metric calculations for 90 trees. The function takes a dataset and a tree (root node) as input and returns the average pre-prune accuracy and average pruned accuracy. Additionally, it prints the confusion matrix, accuracy, recall, precision, F1-score, and depth for both pre-pruned and pruned trees to the console.

3. `depth_search(tree, validation_set, original_tree, depth)`: This function performs the pruning process on a single tree. If you want to prune a tree independently, you can call this function. It takes the tree (root node), a validation set, the original root node, and the depth as input. The function outputs the pruned tree (root node) and its depth.

4. `create_plot(tree, depth)`: This function allows visualizing the decision tree. If you want to draw a tree, you can call this function. To use this function, first call `decision_tree_learning()` to obtain a tree and depth. Then, use those values to call `create_plot()`. The function generates a PNG file named "tree.png" representing the tree structure.

When the main function is called, it shuffles the data and then executes the following steps: `decision_tree_learning()`, `create_plot()`, and `evaluate()`.

To modify the dataset, simply edit the file path in the call to `np.loadtxt()` within the main function.
