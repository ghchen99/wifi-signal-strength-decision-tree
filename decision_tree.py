import numpy as np
from numpy.random import shuffle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def entropy(array):  # Entropy of rooms
    rooms = []
    for row in array:
        rooms.append(row[-1])  # Isolate last column (room no)
    room_number, label_occurrences = np.unique(rooms, return_counts=True)  # room no: 1 2 3 4
    hits = label_occurrences / len(array)  # label_occurrences = no. occurrences of room no: 1 2 3 4 respectvively
    entropy = -(hits * np.log2(hits)).sum()  # entropy = plog2(p)
    return entropy


def find_split(array):  # Finding split/threshold
    entropy_all = entropy(array)  # Entropy of whole dataset (Entropy(S))

    max_change = [0, 0, 0, 0]  # [source_no, index, remainder, midpoint]
    rows, columns = array.shape[0], array.shape[1] - 1

    # Iterate over each column except last column as it holds the room number
    for i in range(columns):
        # Sort array by current column
        sorted_array = array[np.argsort(array[:, i])]

        # Find the points where Room changes value
        for j in range(rows - 1):
            if rows == 2:
                midpoint = (sorted_array[j, i] + sorted_array[j + 1, i]) / 2
                diff = abs(sorted_array[j, i] - sorted_array[j + 1, i])
                if max_change[2] < diff:
                    max_change = [i, j, diff, midpoint]

            elif sorted_array[j, columns] != sorted_array[j + 1, columns]:
                # Take the midpoint of these two values in the current column
                midpoint = (sorted_array[j, i] + sorted_array[j + 1, i]) / 2

                # Find the Gain(midpoint, S)
                remainder = (((j + 1) / rows) * entropy(sorted_array[:j + 1, :])) + (((rows - (j + 1)) / rows) * entropy(sorted_array[j + 1:, :]))
                gain = entropy_all - remainder

                # If Gain > max_change.gain max_change = midpoint, gain
                if gain > max_change[2]:
                    max_change = [i, j, gain, midpoint]
            # Continue until all elements have been read in that column and the max midpoint has been identified
    return max_change[0], max_change[1], max_change[3]


def label_same(array):
    initial = array[0][-1]  # Last element of first row (attribute)
    for row in array:
        if row[-1] != initial:
            return False
    return True


def decision_tree_learning(training, depth):
    if label_same(training):
        node = {
            "attribute": training[0][-1],
            "value": None,
            "left": None,
            "right": None,
            "leaf": True,
        }
        return node, depth

    attribute, index, split_value = find_split(training)

    # Sort data
    sorted_data = training[np.argsort(training[:, attribute])]
    left_set = sorted_data[:index + 1, :]
    right_set = sorted_data[index + 1:, :]
    node = {
        "attribute": attribute,
        "value": split_value,
        "left": None,
        "right": None,
        "leaf": False,
    }

    left_set = np.array(left_set)
    right_set = np.array(right_set)

    node["left"], l_depth = decision_tree_learning(left_set, depth + 1)
    node["right"], r_depth = decision_tree_learning(right_set, depth + 1)
    return node, max(l_depth, r_depth)


def metrics(confusion_matrix, validation):
    # define room 1 as positive i.e. A[0][0]
    true_pos = confusion_matrix[0][0]
    true_neg = confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[3][3]
    false_pos = confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[3][0]
    false_neg = confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3]

    # define room 2 as positive i.e. A[1][1]
    true_pos += confusion_matrix[1][1]
    true_neg += confusion_matrix[0][0] + confusion_matrix[2][2] + confusion_matrix[3][3]
    false_pos += confusion_matrix[0][1] + confusion_matrix[2][1] + confusion_matrix[3][1]
    false_neg += confusion_matrix[1][0] + confusion_matrix[1][2] + confusion_matrix[1][3]

    # define room 3 as positive i.e. A[2][2]
    true_pos += confusion_matrix[2][2]
    true_neg += confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[3][3]
    false_pos += confusion_matrix[0][2] + confusion_matrix[1][2] + confusion_matrix[3][2]
    false_neg += confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][3]

    # define room 3 as positive i.e. A[3][3]
    true_pos += confusion_matrix[3][3]
    true_neg += confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]
    false_pos += confusion_matrix[0][3] + confusion_matrix[1][3] + confusion_matrix[2][3]
    false_neg += confusion_matrix[3][0] + confusion_matrix[3][1] + confusion_matrix[3][2]

    true_pos /= 4
    true_neg /= 4
    false_pos /= 4
    false_neg /= 4

    accuracy = (true_pos + true_neg) / (true_pos+true_neg+false_neg+false_pos)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    F1 = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, F1


def make_confusion(node, validation):
    confusion_matrix = np.zeros(shape=(4, 4))
    for row in validation:
        actual_room = row[-1]
        predicted_room = traverse(node, 0, row)
        confusion_matrix[int(actual_room-1)][int(predicted_room-1)] += 1
    row_sums = confusion_matrix.sum(axis=1)
    norm_matrix = confusion_matrix / row_sums[:,np.newaxis]
    return norm_matrix


def evaluate(all_data, node):
    decile = 0.1 * len(all_data)

    # Minimum row no
    data_min = 0

    # Max row no
    data_max = all_data.shape[0]

    # Max nested row no
    nest_max = int(0.9 * data_max)

    average_preprune_accuracy = 0
    average_prune_accuracy = 0

    average_preprune_precision = 0
    average_prune_precision = 0

    average_preprune_recall = 0
    average_prune_recall = 0

    average_preprune_F1 = 0
    average_prune_F1 = 0

    average_preprune_depth = 0
    average_prune_depth = 0

    average_preprune_matrix = np.zeros(shape=(4, 4))
    average_prune_matrix = np.zeros(shape=(4, 4))

    for i in range(10):
        # increments
        x = int(i * decile)

        # Split into data ranges, A training (80%), B validation (10%), C testing (10%)
        A_start = x
        A_end = int(x + (8*decile))

        B_start = A_end
        B_end = int(B_start + decile)

        C_start = B_end
        C_end = int(C_start + decile)

        if A_start > data_max:
            A_start = A_start - data_max

        if B_start > data_max:
            B_start = B_start - data_max

        if C_start > data_max:
            C_start = C_start - data_max

        if A_end > data_max:
            A_end = A_end - data_max

        if B_end > data_max:
            B_end = B_end - data_max

        if C_end > data_max:
            C_end = C_end - data_max

        if A_end > A_start:
            training = all_data[A_start:A_end]
        else:
            training = np.concatenate([all_data[A_start:data_max], all_data[data_min:A_end]])

        if B_end > B_start:
            validation = all_data[B_start:B_end]
        else:
            validation = np.concatenate([all_data[B_start:data_max], all_data[data_min:B_end]])

        if C_end > C_start:
            testing = all_data[C_start:C_end]
        else:
            testing = np.concatenate([all_data[C_start:data_max], all_data[data_min:C_end]])

        nest_data = np.concatenate((training, validation))

        for j in range(9):
            # We are iterating through the train (u) and eval (v) datasets
            y = int(j * decile)

            U_start = y
            U_end = int(y + (8*decile))

            V_start = U_end
            V_end = int(V_start + decile)

            if U_start > nest_max:
                U_start = U_start - nest_max

            if V_start > nest_max:
                V_start = V_start - nest_max

            if U_end > nest_max:
                U_end = U_end - nest_max

            if V_end > nest_max:
                V_end = V_end - nest_max

            if U_end > U_start:
                training = nest_data[U_start:U_end]
            else:
                training = np.concatenate([nest_data[U_start:nest_max], nest_data[data_min:U_end]])

            if V_end > V_start:
                validation = nest_data[V_start:V_end]
            else:
                validation = np.concatenate([nest_data[V_start:nest_max], nest_data[data_min:V_end]])

            node, depth = decision_tree_learning(training, 0)

            confusion_matrix = make_confusion(node, testing)
            average_preprune_matrix = np.add(confusion_matrix, average_preprune_matrix)
            preprune_accuracy, preprune_precision, preprune_recall, preprune_F1 = metrics(confusion_matrix, testing)

            average_preprune_accuracy += preprune_accuracy
            average_preprune_precision += preprune_precision
            average_preprune_recall += preprune_recall
            average_preprune_F1 += preprune_F1
            average_preprune_depth += depth

            # Finding the pruned accuracy
            pruned_tree, pruned_depth = depth_search(node, training, validation, node, 0)

            pruned_confusion = make_confusion(pruned_tree, testing)
            average_prune_matrix = np.add(pruned_confusion, average_prune_matrix)
            pruned_accuracy, pruned_precision, pruned_recall, pruned_F1 = metrics(pruned_confusion, testing)

            average_prune_accuracy += pruned_accuracy
            average_prune_precision += pruned_precision
            average_prune_recall += pruned_recall
            average_prune_F1 += pruned_F1
            average_prune_depth += pruned_depth

    average_preprune_accuracy /= 90
    average_prune_accuracy /= 90

    average_preprune_F1 /= 90
    average_prune_F1 /= 90

    average_preprune_recall /= 90
    average_prune_recall /= 90

    average_preprune_precision /= 90
    average_prune_precision /= 90

    average_preprune_matrix /= 90
    average_prune_matrix /= 90

    average_preprune_depth /= 90
    average_prune_depth /= 90

    print("average preprune accuracy: ", average_preprune_accuracy)
    print("average prune accuracy: ", average_prune_accuracy)

    print("average preprune recall: ", average_preprune_recall)
    print("average prune recall: ", average_prune_recall)

    print("average preprune precision: ", average_preprune_precision)
    print("average prune precision: ", average_prune_precision)

    print("average preprune F1: ", average_preprune_F1)
    print("average prune F1: ", average_prune_F1)

    print("average preprune depth: ", average_preprune_depth)
    print("average prune depth: ", average_prune_depth)

    print("average preprune confusion matrix: ")
    print(average_preprune_matrix)

    print("average prune confusion matrix: ")
    print(average_prune_matrix)

    return average_preprune_accuracy, average_prune_accuracy


def traverse(node, room, test_row):
    # if node value == none, we are at a leaf node
    if node["value"] is None:
        room = node["attribute"]
    elif test_row[node["attribute"]] < node["value"]:
        # node[attribute] = column being tested, test_row[node[attribute]] = value in test row in that column that we want to compare
        room = traverse(node["left"], room, test_row)
    else:
        room = traverse(node["right"], room, test_row)

    return room


def depth_search(node, training, validation, root_node, depth):
    # we are searching for a parent node with two children that are leaves
    # If we reach a leaf move back up the tree, this has the effect that if a node is pruned, it will be checked again by the function
    if node["value"] is None:
        return node, depth

    elif (node["left"]["leaf"] is True) and (node["right"]["leaf"] is True):
        # Finding accuracy of original node
        confusion_matrix = make_confusion(root_node, validation)
        accuracy = metrics(confusion_matrix, validation)[0]

        # Save the old node incase
        old_node = node.copy()

        # Remove the leaf nodes, set the current node as a leaf
        node["left"] = None
        node["right"] = None
        node["leaf"] = True
        node["value"] = None

        # Find which room has the most occurances set = attribute
        # Need the set of data associated with each node
        # Taking column of rooms only
        rooms = training[:, -1].astype(int)
        frequent_room = np.argmax(np.bincount(rooms.flat))
        node["attribute"] = frequent_room

        # Test the accuracy here
        confusion_matrix = make_confusion(root_node, validation)
        new_accuracy = metrics(confusion_matrix, validation)[0]

        # If accuracy decreased undo prune
        if new_accuracy < accuracy:
            node = old_node.copy()
        # Else do nothing

    else:
        # Keep traversing until reached leaves

        # splitting the data as we move through nodes
        attribute, index, split_value = find_split(training)

        sorted_data = training[np.argsort(training[:, attribute])]
        left_set = sorted_data[:index + 1, :]
        right_set = sorted_data[index + 1:, :]

        # search the left and right branches
        node["left"], l_depth = depth_search(node["left"], left_set, validation, root_node, depth + 1)
        node["right"], r_depth = depth_search(node["right"], right_set, validation, root_node, depth + 1)
        depth = max(l_depth, r_depth)

    return node, depth


def num_leaves(root):
    if root is None:
        return 0
    if(root["left"] is None and root["right"] is None):
        return 1
    return num_leaves(root["left"]) + num_leaves(root["right"])


def print_tree(decision, leaf, arrows, root, root_x, root_y, canvas_w, canvas_h, static_root):  # if the first key tells you what feat was split on

    if (root is None):
        return
    numLeafs = num_leaves(root)  #this determines the x width of this tree

    midpoint_x = print_tree.x_val + (numLeafs/canvas_w)
    midpoint_y = print_tree.y_val

    print_tree.y_val -=10/canvas_h

    if( root['right'] != None and root['left'] != None):
        tree_text= str("[X" + str(root.get("attribute", "Empty")) + " < " + str(root.get("value", "Empty")) + "]")



        print_tree.direction = 'l'
        print_tree(decision, leaf, arrows, root['left'], midpoint_x, midpoint_y - 0.025, canvas_w, canvas_h, static_root)
        print_tree.direction = 'r'
        print_tree(decision, leaf, arrows, root['right'], midpoint_x, midpoint_y - 0.025, canvas_w, canvas_h, static_root)

        if(root == static_root):
            plt.annotate(tree_text, xy=(root_x, root_y), xycoords='axes fraction', xytext=(midpoint_x, midpoint_y), textcoords='axes fraction', va='center', ha='center', bbox=decision)
        else:
            plt.annotate(tree_text, xy=(root_x, root_y), xycoords='axes fraction', xytext=(midpoint_x, midpoint_y), textcoords='axes fraction', va='center', ha='center', bbox=decision, arrowprops=arrows)




    else:
        #This is a leaf node
        tree_text= str("Leaf: " + str(root.get("attribute", "Empty")))

        if(print_tree.direction == 'l'):
            plt.annotate(tree_text, xy=(root_x, root_y), xycoords='axes fraction', xytext=(midpoint_x - 0.05, midpoint_y), textcoords='axes fraction', va='center', ha='center', bbox=leaf, arrowprops=arrows)
        elif(print_tree.direction == 'r'):
            plt.annotate(tree_text, xy=(root_x, root_y), xycoords='axes fraction', xytext=(midpoint_x + 0.05, midpoint_y), textcoords='axes fraction', va='center', ha='center', bbox=leaf, arrowprops=arrows)

    print_tree.x_val += 1.0/canvas_w
    print_tree.y_val += 10/canvas_h


def create_plot(root, depth):

    # Decorations
    decision = dict(boxstyle="round", edgecolor="#023e8a", facecolor="white", )
    leaf = dict(boxstyle="round", facecolor="0.8")
    arrows = dict(arrowstyle="<-", color="#9d0208")

    fig = plt.figure(1, facecolor="white")
    fig.clf()
    #remove grid
    axprops = dict(xticks=[], yticks=[])
    plt.subplot(111, frameon=False, **axprops)

    canvas_w = float(0.04*num_leaves(root))
    canvas_h = float(depth)
    print_tree.x_val=0.5/canvas_w
    print_tree.y_val=1.0
    root_x = 0.5
    root_y = 1.0

    static_root = root

    print_tree(decision, leaf, arrows, root, root_x, root_y, canvas_w, canvas_h, static_root)

    plt.savefig("treeclean.png", bbox_inches="tight")


def main():

    all_data = np.loadtxt("WIFI.db/clean_dataset.txt")

    shuffle(all_data)

    decile = 0.1 * len(all_data)
    training_number = int(8 * decile)
    validation_number = int(decile)
    training = all_data[:training_number]
    validation = all_data[training_number:(training_number+validation_number)]
    node, depth = decision_tree_learning(training, 0)

    print("-----PRINT TREE------")
    create_plot(node, depth)

    # print(node)
    print("Number of Leaves: ", num_leaves(node))

    evaluate(all_data, node)


main()
