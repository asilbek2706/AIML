import numpy as np
import pandas as pd

# Read the dataset
data = pd.read_csv("./data/training_data.csv")

# Separate attributes and target column
concepts = np.array(data.iloc[:, :-1], dtype=str)
target = np.array(data.iloc[:, -1], dtype=str)


def learn(concepts, target):
    # Initialize S from the first positive example
    positive_indices = [
        i for i, label in enumerate(target) if label.strip().lower() == "yes"
    ]

    if not positive_indices:
        raise ValueError("The dataset must contain a positive example (yes).")

    first_positive = positive_indices[0]
    specific_h = concepts[first_positive].copy()

    # Initialize G with the maximally general hypothesis
    general_h = [["?" for _ in range(len(specific_h))]]

    print("Initial Specific Hypothesis S:", specific_h)
    print("Initial General Hypothesis G:", general_h)

    for i, example in enumerate(concepts):
        label = target[i].strip().lower()

        if label == "yes":
            # Generalize S to cover the positive example
            for j in range(len(specific_h)):
                if specific_h[j] != example[j]:
                    specific_h[j] = "?"

            # Remove G hypotheses that do not cover this positive example
            general_h = [
                g
                for g in general_h
                if all(
                    g[j] == "?" or g[j] == example[j]
                    for j in range(len(example))
                )
            ]

        elif label == "no":
            # Specialize G to exclude the negative example
            new_general = []

            for g in general_h:
                # Agar g misolni qamrab olsa (cover qilsa), uni ixtisoslashtiramiz
                if all(
                    g[j] == "?" or g[j] == example[j]
                    for j in range(len(example))
                ):
                    for j in range(len(specific_h)):
                        if (
                            specific_h[j] != "?"
                            and example[j] != specific_h[j]
                        ):
                            specialization = g.copy()
                            specialization[j] = specific_h[j]
                            new_general.append(specialization)
                else:
                    # Agar g misolni qamrab olmasa, o'zini qoldiramiz
                    new_general.append(g)

            general_h = new_general

        else:
            raise ValueError(
                f"Unknown target label at row {i + 2}: {target[i]}"
            )

        print(f"\nStep {i + 1}: {label.upper()} example")
        print("S =", specific_h)
        print("G =", general_h)

        # Remove duplicate hypotheses
        general_h = [
            list(g) for g in dict.fromkeys(tuple(g) for g in general_h)
        ]

    return specific_h, general_h


s_final, g_final = learn(concepts, target)

print("\nFinal Specific Hypothesis S:")
print(s_final)

print("\nFinal General Hypothesis G:")
print(g_final)