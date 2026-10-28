import math
import pandas as pd

def entropy(column):
  """Berilgan ustun uchun entropiyani hisoblaydi."""
  counts = column.value_counts().values
  total = sum(counts)
  ent = 0
  for c in counts:
    p = c / total
    ent -= p * math.log2(p)
  return ent

def information_gain(df, attribute, target):
  """Berilgan atribut uchun axborot olishni (Information Gain) hisoblaydi."""
  total_entropy = entropy(df[target])
  values = df[attribute].unique()
  weighted_entropy = 0

  for val in values:
    subset = df[df[attribute] == val]
    weight = len(subset) / len(df)
    weighted_entropy += weight * entropy(subset[target])

  return total_entropy - weighted_entropy

def id3(df, target, attributes):
  """ID3 algoritmi yordamida qarorlar daraxtini (Decision Tree) quradi."""
  # Baza holatlari (Base cases)
  if len(df[target].unique()) == 1:
    return df[target].iloc[0]
  if len(attributes) == 0:
    return df[target].mode()[0]

  # Eng yaxshi atributni tanlash
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


# --- Namuna Ma'lumotlar (Dataset) ---
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain"],
    "Humidity": ["Normal", "Normal", "High", "High", "Normal"],
    "Play": ["No", "Yes", "Yes", "Yes", "Yes"],
}
df = pd.DataFrame(data)

# --- ID3 Algoritmini ishga tushirish ---
attributes = ["Outlook", "Humidity"]
decision_tree = id3(df, "Play", attributes)

print("Decision Tree:")
print(decision_tree)