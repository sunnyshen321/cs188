import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """

        while True:
            count = 0
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) == nn.as_scalar(y):
                     continue
                count += 1
                nn.Parameter.update(self.get_weights(), x, nn.as_scalar(y))
            if count == 0:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        self.lr = .004
        self.b1 = nn.Parameter(1, 100)
        self.w1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 1)
        #self.w3 = nn.Parameter(1, 1)
        self.batch = 5

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        relu = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        predicted_y = nn.Linear(relu, self.w2)
        #predicted_y = nn.Linear(relu2, self.w3)

        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """

        while True:
            for x,y in dataset.iterate_once(self.batch):
                loss = self.get_loss(x, y)
                gradient = nn.gradients(loss, [self.w1, self.b1, self.w2])
                self.w1.update(gradient[0], -self.lr)
                self.b1.update(gradient[1], -self.lr)
                self.w2.update(gradient[2], -self.lr)
                #self.w3.update(gradient[3], -self.lr)

            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < .02:
                break







class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        self.lr = .13
        self.b1 = nn.Parameter(1, 300)
        self.w1 = nn.Parameter(784, 300)
        self.w2 = nn.Parameter(300, 10)
        self.batch = 10

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        relu = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        predicted_y = nn.Linear(relu, self.w2)

        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for x,y in dataset.iterate_once(self.batch):
                loss = self.get_loss(x, y)
                gradient = nn.gradients(loss, [self.w1, self.b1, self.w2])
                self.w1.update(gradient[0], -self.lr)
                self.b1.update(gradient[1], -self.lr)
                self.w2.update(gradient[2], -self.lr)
                #self.w3.update(gradient[3], -self.lr)

            if dataset.get_validation_accuracy() >= .98:
                break
            self.lr *= .8

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.lr = .12
        self.b1 = nn.Parameter(1, 200)
        self.w1 = nn.Parameter(self.num_chars, 200)

        self.w_h1 = nn.Parameter(200, 200)
        #self.w_h2 = nn.Parameter(200, 200)
        self.b_h1 = nn.Parameter(1, 200)
        #self.b_h2 = nn.Parameter(1, 200)

        self.batch = 30

        self.out = nn.Parameter(200, len(self.languages))


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        #initialize h
        h = nn.AddBias(nn.Linear(xs[0], self.w1), self.b1)

        for x in xs[1:]:
            # x: shape batch_size x self.num_chars
            temp = nn.Add(nn.Linear(x, self.w1), nn.Linear(h, self.w_h1))
            #temp2 = nn.AddBias(nn.Add(temp, nn.Linear(h, self.w_h2)), self.b_h1)
            h = nn.ReLU(nn.AddBias(temp, self.b_h1))

        return nn.Linear(h, self.out)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for x,y in dataset.iterate_once(self.batch):
                loss = self.get_loss(x, y)
                gradient = nn.gradients(loss, [self.w1, self.b1, self.w_h1, self.b_h1, self.out])
                self.w1.update(gradient[0], -self.lr)
                self.b1.update(gradient[1], -self.lr)
                self.w_h1.update(gradient[2], -self.lr)
                #self.w_h2.update(gradient[3], -self.lr)
                self.b_h1.update(gradient[3], -self.lr)
                #self.b_h2.update(gradient[5], -self.lr)
                self.out.update(gradient[4], -self.lr)

            if dataset.get_validation_accuracy() >= .9:
                break

            self.lr *= .97

