"""

Solution outline for the COM2004/3004 assignment.

Author: Krish Panchigar
"""
from typing import List, Any

import numpy as np
from scipy.linalg import eigh
from scipy.stats import multivariate_normal
from scipy.signal import gaussian, convolve2d

N_DIMENSIONS = 10

# TODO: replace multiline comments with """
# TODO: add docstrings to functions


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """ Uses Gaussian distribution to calculate the probability of each square being each label
    Args:
        train (np.ndarray): training data feature vectors stored as rows.
        train_labels (np.ndarray): the labels corresponding to the feature vectors.
        test (np.ndarray): test data feature vectors stored as rows.
    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    unique_labels = np.unique(train_labels)

    probabilities = []

    for label in unique_labels:
        label_train = train[train_labels == label]
        mean = np.mean(label_train, axis=0)
        cov = np.cov(label_train, rowvar=False)
        distribution = multivariate_normal(mean, cov)
        p = distribution.pdf(test)
        probabilities.append(p)

    probabilities = np.array(probabilities)

    labels = unique_labels[np.argmax(probabilities, axis=0)]
    return labels.tolist()


# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """ Uses the eigenvectors from the PCA on the test data to reduce the dimensionality of the data

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data which stores the eigenvectors.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    v = np.array(model["eigenvectors"])

    pca_data = np.dot((data - np.mean(data)), v)

    return pca_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.
    Calculates the eigenvectors of the covariance matrix of the training data and stores them in the model.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    unique_labels = np.unique(labels_train)

    covx = np.cov(fvectors_train, rowvar=False)
    N = covx.shape[0]
    w, v = eigh(covx, eigvals=(N - 10, N - 1))
    v = np.fliplr(v)
    model = {}
    model["eigenvectors"] = v.tolist()
    model["labels_train"] = labels_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """
    Applies a gaussian filter to the image to reduce noise and crops the image to remove the border.
    Args:
        images:

    Returns: an array of feature vectors

    """
    h, w = images[0].shape
    n_features = (h - 6) * (w - 6)
    fvectors = np.empty((len(images), n_features))

    kernel_size = 2
    sigma = 5
    kernel = gaussian(kernel_size, sigma).reshape(kernel_size, 1)
    kernel = np.dot(kernel, kernel.transpose())

    for i, image in enumerate(images):
        # Apply Gaussian filter
        image = convolve2d(image, kernel, mode='same', boundary='symm')
        # Crop the image
        cropped_image = image[3:-3, 3:-3]
        fvectors[i, :] = cropped_image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.
    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def get_square_probabilities(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> list[tuple[Any, Any]]:
    """Get the probabilities for all the labels for a square.
    Args:
        train (np.ndarray): training data feature vectors stored as rows.
        train_labels (np.ndarray): the labels corresponding to the feature vectors.
        test (np.ndarray): the feature vector of the square to get the probabilities for.

    Returns:
        A list of tuples containing the probability and the label for each label.
    """
    unique_labels = np.unique(train_labels)
    probabilities = []
    for label in unique_labels:
        label_train = train[train_labels == label]
        mean = np.mean(label_train, axis=0)
        cov = np.cov(label_train, rowvar=False)
        distribution = multivariate_normal(mean, cov)
        p = distribution.pdf(test)
        probabilities.append((p, label))

    return probabilities


def get_n_highest_probability_label(square_index, model, fvectors_test, n):
    """
    Get the label with the nth-highest (0 indexed)  probability for a square.
    Args:
        square_index: index of the square in the feature vector array
        model: the model data
        fvectors_test: the feature vector array
        n: the nth-highest probability to get

    Returns:
        the label with the nth-highest probability
    """
    square_probabilities = get_square_probabilities(
        np.array(model["fvectors_train"]),
        np.array(model["labels_train"]),
        fvectors_test[square_index]
    )
    sorted_probabilities = sorted(square_probabilities, key=lambda x: x[0], reverse=True)
    highest_prob, highest_label = sorted_probabilities[n]
    return highest_label


def check_pawns_on_rank(rank: int, pawn_label: str, board: np.ndarray, model: dict, fvectors_test: np.ndarray, i: int):
    """
    Check if there are pawns on the first or last rank and replace them with the second-highest probability label.
    Args:
        rank: the rank to check: 0 or 7
        pawn_label: the label of the pawns to check: "P" or "p"
        board: the board to check
        model: the model data
        fvectors_test: the feature vector array
        i: the index of the board in the boards array
    Returns:
        the board with the pawns replaced with the second-highest probability label
    """
    if pawn_label in board[rank]:
        invalid_pawns = np.where(board[rank] == pawn_label)
        for j in range(len(invalid_pawns[0])):
            pawn = (rank, invalid_pawns[0][j])
            pawn_index = pawn[0] * 8 + pawn[1] + (i * 64)
            second_highest_label = get_n_highest_probability_label(pawn_index, model, fvectors_test, 1)
            board[pawn] = second_highest_label
    return board


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.
    Checks the validity of the board and replaces invalid squares with the second-highest probability label.
    Checks for pawns on the first or last rank and replaces them with the second-highest probability label.
    Checks for more than 8 pawns of one colour on the board and replaces the lowest probability pawns with the
    second-highest probability label.
    Checks for no kings of either colour on the board and sets the square with the highest probability of being
    the king to a king.
    Checks if there are more than one king of either colour on the board and sets the square with the highest
    probability of being the king to a king and the other kings to the second-highest probability label.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    labels = np.array(classify_squares(fvectors_test, model))
    n_boards = int(len(labels) / 64)
    boards = labels.reshape(n_boards, 8, 8)

    # check validity of boards
    for i in range(len(boards)):
        board = boards[i]
        kings = ["K", "k"]
        for king_label in kings:
            # more than one king of one colour on the board
            if np.sum(board == king_label) > 1:
                invalid_kings = np.where(board == king_label)
                highest_prob_king = None
                highest_prob = 0
                for j in range(len(invalid_kings[0])):
                    king = (invalid_kings[0][j], invalid_kings[1][j])
                    king_index = king[0] * 8 + king[1] + (i * 64)
                    king_prob = get_square_probabilities(np.array(model["fvectors_train"]),
                                                         np.array(model["labels_train"]),
                                                         fvectors_test[king_index])[2 if king_label == 'K' else 8][0]
                    if king_prob > highest_prob:
                        highest_prob = king_prob
                        highest_prob_king = king

                board[highest_prob_king] = king_label
                # set the other king to be second-highest probability
                for k in range(len(invalid_kings[0])):
                    king = (invalid_kings[0][k], invalid_kings[1][k])
                    if king != highest_prob_king:
                        king_index = king[0] * 8 + king[1] + (i * 64)
                        second_highest_label = get_n_highest_probability_label(king_index, model, fvectors_test, 1)
                        board[king] = second_highest_label
                boards[i] = board

            # no kings on the board
            if np.sum(board == king_label) == 0:
                # Get the probabilities for each square being a king
                king_probabilities = []
                for j in range(8):
                    for k in range(8):
                        square = (j, k)
                        square_index = square[0] * 8 + square[1] + (i * 64)
                        square_prob = get_square_probabilities(np.array(model["fvectors_train"]),
                                                               np.array(model["labels_train"]),
                                                               fvectors_test[square_index])[2 if king_label == 'K' else 8][0]
                        king_probabilities.append((square_prob, square))

                sorted_squares = sorted(king_probabilities, key=lambda x: x[0], reverse=True)
                # Get the square with the highest probability
                best_square = sorted_squares[0][1]
                board[best_square] = king_label
            boards[i] = board

        # check for pawns on first or last rank and replace with second-highest probability
        board = check_pawns_on_rank(0, "P", board, model, fvectors_test, i)
        board = check_pawns_on_rank(7, "P", board, model, fvectors_test, i)
        board = check_pawns_on_rank(0, "p", board, model, fvectors_test, i)
        board = check_pawns_on_rank(7, "p", board, model, fvectors_test, i)

        # more than 8 pawns of one colour on the board
        for pawn_label in ['P', 'p']:
            if np.sum(board == pawn_label) > 8:
                invalid_pawns = np.where(board == pawn_label)

                # Get the probabilities for each pawn
                pawn_probabilities = []
                for j in range(len(invalid_pawns[0])):
                    pawn = (invalid_pawns[0][j], invalid_pawns[1][j])
                    pawn_index = pawn[0] * 8 + pawn[1] + (i * 64)
                    pawn_prob = get_square_probabilities(np.array(model["fvectors_train"]),
                                                         np.array(model["labels_train"]),
                                                         fvectors_test[pawn_index])[2 if pawn_label == 'P' else 8][0]
                    pawn_probabilities.append((pawn_prob, pawn))

                # Sort the pawns by their probabilities in descending order
                sorted_pawns = sorted(pawn_probabilities, key=lambda x: x[0], reverse=True)

                # Get the lowest probability pawns
                rest_pawns = sorted_pawns[8:]

                # get the second-highest probability label and update the square with this label
                for pawn_prob, pawn in rest_pawns:
                    pawn_index = pawn[0] * 8 + pawn[1] + (i * 64)
                    second_highest_label = get_n_highest_probability_label(pawn_index, model, fvectors_test, 1)
                    board[pawn] = second_highest_label

            boards[i] = board

            # no kings on the board
    return boards.flatten().tolist()
