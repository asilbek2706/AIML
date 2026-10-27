import pandas as pd
import math

def entropy(column):
    values, counts = column.value_counts().index, column.value_counts().values
    total = sum(counts)
    ent = 0
    for c in counts:
        p = c / total
    ent -= p * math.log2(p)
    return ent

def information_gain(df, attribute, target):
    global subset
    total_entropy = entropy(df[target])
    values = df[attribute].unique()
    weighted_entropy = 0

    for val in values:
        subset = df[df[attribute] == val]
    weight = len(subset) / len(df)
    weighted_entropy += weight * entropy(subset[target])

    return total_entropy - weighted_entropy

def id3(df, target, attributes):
    # Base cases
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if len(attributes) == 0:
        return df[target].mode()[0]

    # Choose best attribute
    gains = {attr: information_gain(df, attr, target) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}

    for val in df[best_attr].unique():
        subset = df[df[best_attr] == val]

        if subset.empty:
            tree[best_attr][val] = df[target].mode()[0]
        else:
            new_attrs = [a for a in attributes if a != best_attr]
            tree[best_attr][val] = id3(subset, target, new_attrs)

            return tree
    return None

data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain"],
    "Humidity": ["High", "Normal", "High", "High", "Normal"],
    "Play": ["No", "Yes", "Yes", "Yes", "No"]
}
df = pd.DataFrame(data)

attributes = ["Outlook", "Humidity"]
decision_tree = id3(df, "Play", attributes)
print("Decision Tree:")
print(decision_tree)
