import math
from collections import defaultdict

class BernoulliNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_prob = {}
        self.classes = []
        self.n_features = 0

    def fit(self, X, y):
        self.classes = set(y)
        self.n_features = len(X[0])

        class_count = defaultdict(int)
        feature_count = defaultdict(lambda: [0] * self.n_features)

        for xi, yi in zip(X, y):
            class_count[yi] += 1
            for j in range(self.n_features):
                if xi[j] == 1:
                    feature_count[yi][j] += 1

        total = len(y)

        # Class prior probability
        self.class_priors = {
            c: class_count[c] / total for c in self.classes
        }

        # Feature probability with Laplace smoothing
        self.feature_prob = {}
        for c in self.classes:
            self.feature_prob[c] = []
            for j in range(self.n_features):
                p = (feature_count[c][j] + 1) / (class_count[c] + 2)
                self.feature_prob[c].append(p)

    def predict_one(self, x):
        best_class = None
        best_log_prob = -float("inf")

        for c in self.classes:
            log_prob = math.log(self.class_priors[c])
            for j in range(self.n_features):
                p = self.feature_prob[c][j]
                if x[j] == 1:
                    log_prob += math.log(p)
                else:
                    log_prob += math.log(1 - p)

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c

        return best_class

    def predict(self, X):
        return [self.predict_one(x) for x in X]


# Example usage
if __name__ == "__main__":
    # Binary features
    X_train = [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 0]
    ]
    y_train = [1, 1, 0, 0, 1]

    model = BernoulliNaiveBayes()
    model.fit(X_train, y_train)

    X_test = [
        [1, 0, 0],
        [0, 0, 1],
        [1, 1, 0]
    ]

    predictions = model.predict(X_test)
    print("Predictions:", predictions)
