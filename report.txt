# Chessboard Diagram Assignment Report

## Feature Extraction

Firstly, I apply a Gaussian filter to reduce noise and enhance key features. Then I crop the image of the square to
remove the 3 bordering rows and columns. This removes some of the irrelevant data like the pixels of the square that
don't contain any information about the piece. I then perform PCA on the image to get the 10 most important features
from the training data. This technique allows the capture of the most important aspects of the data by transforming it
into a lower-dimensional space.
I tried using max pooling and average pooling, which takes a grid (2x2 in this case) of pixels and selects the
maximum/average of those pixels in place of those pixels, reducing the image to a 25x25 pixel image before
running PCA on it, but I found that they did not perform as well as cropping the image.

## Square Classifier

The square classifier design incorporates a parametric approach, specifically utilizing a Gaussian distribution model.
I chose this approach because of its ability to capture intricate relationships within the data, providing a
probabilistic framework for classification.
I experimented with the nearest neighbor and K-nearest neighbor approaches, which did not perform as well as the
Gaussian distribution model. The K-nearest neighbor, while performing slightly better on the clean data, did not
perform as well on the noisy data.

## Full-board Classification

The full board classification is done by first classifying each square and then checking each board for validity.
It checks if there is exactly one king of each colour. If there are more than one king pieces of a colour, it keeps the
king with the highest probability of being a king as the king, and replaces the rest of the kings with their
second-best label. If there are no kings of a colour, it checks each square for the probability of being a king,
and sets the square with the highest probability to be a king.
It also checks that there are no pawns on the first or last row. If there are, it replaces them with the second most
likely label.
It also checks that there are no more than 8 pawns of each colour. If there are, it replaces the pawns with the
lowest probability of being a pawn with the second most likely label.

## Performance

My percentage correctness scores (to 1 decimal place) for the development data are as follows.

High quality data:

- Percentage Squares Correct: 98.9%
- Percentage Boards Correct: 99.1%

Noisy data:

- Percentage Squares Correct: 97.1%
- Percentage Boards Correct: 97.3%
