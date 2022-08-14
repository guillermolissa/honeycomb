from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
from sklearn.metrics import roc_curve, precision_recall_curve, log_loss
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae  # noqa
from sklearn.metrics import r2_score as r2              # noqa

import numpy as np
import math
from matplotlib import pyplot as plt

def true_positive(y_true, y_pred): 
	"""
	Function to calculate True Positives 
    :param y_true: list of true values 
	:param y_pred: list of predicted values 
    :return: number of true positives
	"""
	# initialize
	tp = 0
	for yt, yp in zip(y_true, y_pred):
	    if yt == 1 and yp == 1: 
		    tp += 1
	return tp

def true_negative(y_true, y_pred): 
    """
    Function to calculate True Negatives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of true negatives
    """
    # initialize
    tn = 0
    for yt, yp in zip(y_true, y_pred):
	    if yt == 0 and yp == 0: 
		    tn += 1
    return tn

def false_positive(y_true, y_pred): 
    """
    Function to calculate False Positives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of false positives
    """
	# initialize
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1: 
            fp += 1
    return fp

def false_negative(y_true, y_pred): 
	"""
	Function to calculate False Negatives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of false negatives
	"""
	# initialize
	fn = 0
	for yt, yp in zip(y_true, y_pred):
		if yt == 1 and yp == 0: 
			fn += 1
	return fn

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def mcc(y_true, y_pred):
    """
    This function calculates Matthew's Correlation Coefficient for binary classification.
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: mcc score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    numerator = (tp* tn) - (fp * fn)

    denominator = ((tp + fp)*(fn + tn)*(fp + tn)*(tp + fn))

    denominator = denominator ** 0.5

    return numerator/denominator


def logloss(y, p):
    """Bounded log loss error.

    Args:
        y (numpy.array): target
        p (numpy.array): prediction

    Returns:
        bounded log loss error
    """

    p[p < EPS] = EPS
    p[p > 1 - EPS] = 1 - EPS
    return log_loss(y, p)


def plot_roc_curve(y, p):
    fpr, tpr, _ = roc_curve(y, p)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def plot_pr_curve(y, p):
    precision, recall, _ = precision_recall_curve(y, p)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])




def psi(score_initial, score_new, num_bins = 10, mode = 'fixed'):
    
    eps = 1e-4
    
    # Sort the data
    score_initial.sort()
    score_new.sort()
    
    # Prepare the bins
    min_val = min(min(score_initial), min(score_new))
    max_val = max(max(score_initial), max(score_new))
    if mode == 'fixed':
        bins = [min_val + (max_val - min_val)*(i)/num_bins for i in range(num_bins+1)]
    elif mode == 'quantile':
        bins = pd.qcut(score_initial, q = num_bins, retbins = True)[1] # Create the quantiles based on the initial population
    else:
        raise ValueError(f"Mode \'{mode}\' not recognized. Your options are \'fixed\' and \'quantile\'")
    bins[0] = min_val - eps # Correct the lower boundary
    bins[-1] = max_val + eps # Correct the higher boundary
        
        
    # Bucketize the initial population and count the sample inside each bucket
    bins_initial = pd.cut(score_initial, bins = bins, labels = range(1,num_bins+1))
    df_initial = pd.DataFrame({'initial': score_initial, 'bin': bins_initial})
    grp_initial = df_initial.groupby('bin').count()
    grp_initial['percent_initial'] = grp_initial['initial'] / sum(grp_initial['initial'])
    
    # Bucketize the new population and count the sample inside each bucket
    bins_new = pd.cut(score_new, bins = bins, labels = range(1,num_bins+1))
    df_new = pd.DataFrame({'new': score_new, 'bin': bins_new})
    grp_new = df_new.groupby('bin').count()
    grp_new['percent_new'] = grp_new['new'] / sum(grp_new['new'])
    
    # Compare the bins to calculate PSI
    psi_df = grp_initial.join(grp_new, on = "bin", how = "inner")
    
    # Add a small value for when the percent is zero
    psi_df['percent_initial'] = psi_df['percent_initial'].apply(lambda x: eps if x == 0 else x)
    psi_df['percent_new'] = psi_df['percent_new'].apply(lambda x: eps if x == 0 else x)
    
    # Calculate the psi
    psi_df['psi'] = (psi_df['percent_initial'] - psi_df['percent_new']) * np.log(psi_df['percent_initial'] / psi_df['percent_new'])
    
    # Return the psi values
    return psi_df['psi'].values


def plot_lift(y_real, y_proba, ax = None, color = 'b', title = 'Lift Curve'):
    # Prepare the data
    aux_df = pd.DataFrame()
    aux_df['y_real'] = y_real
    aux_df['y_proba'] = y_proba
    # Sort by predicted probability
    aux_df = aux_df.sort_values('y_proba', ascending = False)
    # Find the total positive ratio of the whole dataset
    total_positive_ratio = sum(aux_df['y_real'] == 1) / aux_df.shape[0]
    # For each line of data, get the ratio of positives of the given subset and calculate the lift
    lift_values = []
    for i in aux_df.index:
        threshold = aux_df.loc[i]['y_proba']
        subset = aux_df[aux_df['y_proba'] >= threshold]
        subset_positive_ratio = sum(subset['y_real'] == 1) / subset.shape[0]
        lift = subset_positive_ratio / total_positive_ratio
        lift_values.append(lift)
    # Plot the lift curve
    if ax == None:
        ax = plt.axes()
    ax.set_xlabel('Proportion of sample')
    ax.set_ylabel('Lift')
    ax.set_title(title)
    sns.lineplot(x = [x/len(lift_values) for x in range(len(lift_values))], y = lift_values, ax = ax, color = color)
    ax.axhline(1, color = 'gray', linestyle = 'dashed', linewidth = 3)




def mape(y, p):
    """Mean Absolute Percentage Error (MAPE).

    Args:
        y (numpy.array): target
        p (numpy.array): prediction

    Returns:
        e (numpy.float64): MAPE
    """

    filt = np.abs(y) > EPS
    return np.mean(np.abs(1 - p[filt] / y[filt]))


def rmse(y_true, y_pred):
    """Root Mean Squared Error (RMSE).

    This function calculates Root mean square error for regression.
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: root mean square error
    """

    # check and get number of samples
    assert y.shape == p.shape

    return np.sqrt(mse(y_true, y_pred))



def gini(y, p):
    """Normalized Gini Coefficient.

    Args:
        y (numpy.array): target
        p (numpy.array): prediction

    Returns:
        e (numpy.float64): normalized Gini coefficient
    """

    # check and get number of samples
    assert y.shape == p.shape

    n_samples = y.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y, p]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    l_true = np.cumsum(true_order) / np.sum(true_order)
    l_pred = np.cumsum(pred_order) / np.sum(pred_order)
    l_ones = np.linspace(1/n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    g_true = np.sum(l_ones - l_true)
    g_pred = np.sum(l_ones - l_pred)

    # normalize to true Gini coefficient
    return g_pred / g_true


# Outcome should be a binary list of the ordinal outcome. [0, 1, 0] for exmaple.
# Probs should be a list of probabilities. [0.79, 0.09, 0.12] for example.
# Outcome and Probs must be provided with the same order as probabilities.

def rps(probs, outcome):
    cum_probs = np.cumsum(probs)
    cum_outcomes = np.cumsum(outcome)
    
    
    sum_rps = 0
    for i in range(len(outcome)):         
        sum_rps+= (cum_probs[i] - cum_outcomes[i])**2
    
    return sum_rps/(len(outcome)-1)