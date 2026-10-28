import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------
# Activation Functions
# ----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# ----------------------------
# Neural Network Class
# ----------------------------
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Weight initialization (using normal or scaled uniform helps stability)
        self.W1 = np.random.randn(self.input_nodes, self.hidden_nodes) * 0.1
        self.W2 = np.random.randn(self.hidden_nodes, self.output_nodes) * 0.1

    def forward(self, X):
        self.hidden_input = np.dot(X, self.W1)
        self.hidden_output = sigmoid(self.hidden_input)

        # Output layer
        self.final_input = np.dot(self.hidden_output, self.W2)
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y):
        # Calculate error
        error = y - self.final_output
        # Gradient for output layer
        d_output = error * sigmoid_derivative(self.final_output)

        # Error for hidden layer
        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        # Update weights
        self.W2 += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.W1 += X.T.dot(d_hidden) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)

            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)


# ----------------------------
# Load and Preprocess Data
# ----------------------------
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode target
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------
# Train the Neural Network
# ----------------------------
nn = NeuralNetwork(
    input_nodes=4,  # iris features
    hidden_nodes=6,  # arbitrary choice
    output_nodes=3,  # 3 classes
    learning_rate=0.1
)

nn.train(X_train, y_train, epochs=2000)

# ----------------------------
# Test the Neural Network
# ----------------------------
predictions = nn.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test_labels) * 100
print(f"\nFinal Test Accuracy = {accuracy:.2f}%")