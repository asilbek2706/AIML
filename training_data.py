import csv

filename = "./data/training_data.csv"

with open(filename, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    data = list(reader)

attributes = data[0][:-1]
training_examples = data[1:]

hypothesis = ["0"] * len(attributes)

print("Initial hypothesis:", hypothesis)

for example in training_examples:
    inputs = example[:-1]
    target = example[-1].strip().lower()

if target == "yes":
    for i in range(len(hypothesis)):
        if hypothesis[i] == "0":
            hypothesis[i] = inputs[i]
        elif hypothesis[i] != inputs[i]:
            hypothesis[i] = "?"

    print("Positive example:", inputs)
    print("Updated hypothesis:", hypothesis)

print("\nFinal most specific hypothesis:")
print(dict(zip(attributes, hypothesis)))
