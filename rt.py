
import json

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shlex
import math

from io import StringIO 
from matplotlib import rcParams
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecessary plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()


def set_display():
    pd.set_option('display.width', 500)
    pd.set_option('display.max_columns', 30)

    #these colors come from colorbrewer2.org. Each is an RGB triplet
    dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                    (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                    (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                    (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                    (0.4, 0.6509803921568628, 0.11764705882352941),
                    (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                    (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                    (0.4, 0.4, 0.4)]

    rcParams['figure.figsize'] = (10, 6)
    rcParams['figure.dpi'] = 150
    rcParams['axes.color_cycle'] = dark2_colors
    rcParams['lines.linewidth'] = 2
    rcParams['axes.grid'] = False
    rcParams['axes.facecolor'] = 'white'
    rcParams['font.size'] = 14
    rcParams['patch.edgecolor'] = 'none'


# get imdb id from movies dataframe
def get_id(dataframe, row):
    tmp = dataframe['imdbID'].irow(row)
    final_id = "%07d" % tmp 
    return final_id
# get rtid id from rotten tomatoes API


def get_rtid(movie_id):
    url = "http://api.rottentomatoes.com/api/public/v1.0/movie_alias.json?id=%s&type=imdb&apikey=xwcad7uj5uqtze4enx4mrtkk" % movie_id
    options = {'id': movie_id, 'type':'imdb', 'apikey': api_key}
    data = requests.get(url,params = options).text
    data = json.loads(data)
    try:
        rtid = json.dumps(data['id'])
    except (KeyError):
        return None
    return rtid


def get_title(dataframe, row):
    return dataframe['title'].irow(row)


#get reviews from rotten tomatoes API
def fetch_reviews(dataframe, row):

    """
    Function
    --------
    fetch_reviews(movies, row)

    Use the Rotten Tomatoes web API to fetch reviews for a particular movie

    Parameters
    ----------
    movies : DataFrame 
      The movies data above
    row : int
      The row of the movies DataFrame to use
      
    Returns
    -------
    If you can match the IMDB id to a Rotten Tomatoes ID:
      A DataFrame, containing the first 20 Top Critic reviews 
      for the movie. If a movie has less than 20 total reviews, return them all.
      This should have the following columns:
        critic : Name of the critic
        fresh  : 'fresh' or 'rotten'
        imdb   : IMDB id for the movie
        publication: Publication that the critic writes for
        quote  : string containing the movie review quote
        review_data: Date of review
        rtid   : Rotten Tomatoes ID for the movie
        title  : Name of the movie
    """


    movie_id = get_id(dataframe, row)
    api_key = 'xwcad7uj5uqtze4enx4mrtkk'
    
    rtid = get_rtid(movie_id)
    url = 'http://api.rottentomatoes.com/api/public/v1.0/movies/%s/reviews.json' % rtid
    ti = get_title(dataframe, row)
    
    # check to make sure rtid for a certain movie exists before you make get request
    if rtid is not None:
        options = {'review_type': 'top_critic', 'page_limit': 20, 'apikey': api_key}
        data = requests.get(url, params=options).text
        data = json.loads(data)  # load a json string into a collection of lists and dicts
        
        if not 'reviews' in data.keys():
            return None
        else:
            if len(data['reviews']) >= 20:
                length = 20
            else:
                length = len(data['reviews'])    

        critic = []
        fresh = []
        imdb = []
        publication = []
        quote = []
        review_date = []
        title = []
        rt= []
        
        #collect data and format to dataframe
        for n in range(length):
            critic.append(json.dumps(data['reviews'][n]['critic'], indent=2)[1:-1])
            fresh.append(json.dumps(data['reviews'][n]['freshness'], indent=2)[1:-1])
            publication.append(json.dumps(data['reviews'][n]['publication'], indent=2)[1:-1])
            quote.append(json.dumps(data['reviews'][n]['quote'], indent=2)[1:-1])
            review_date.append(json.dumps(data['reviews'][n]['date'], indent=2)[1:-1])
            imdb.append(movie_id)
            rt.append(rtid)
            title.append(ti)
    
        reviews = pd.DataFrame({'critic':critic,'fresh':fresh, 'imdb':imdb, 'publication':publication, 'quote':quote, 'review_date':review_date, 'rtid':rt, 'title':title})

        return reviews
    else:
        return None


def build_table(dataframe, rows):
    """
    Function
    --------
    build_table

    Parameters
    ----------
    movies : DataFrame
      The movies data above
    rows : int
      The number of rows to extract reviews for
      
    Returns
    --------
    A dataframe
      The data obtained by repeatedly calling `fetch_reviews` on the first `rows`
      of `movies`, discarding the `None`s,
      and concatenating the results into a single DataFrame
    """
    table = fetch_reviews(movies, 0)
    
    for n in range(1, rows):
        temp = None
        temp = fetch_reviews(movies,n)
        if temp is None:
            continue
        else:
            table = table.append(temp, ignore_index=True)
    return table


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
    return


def plot_reviews(dataframe):
    # plot distribution of review per reviewer
    n = dataframe.size()
    n = n.tolist()
    plt.hist(n, bins = 20)
    remove_border(top=False, right=False)
    plt.ylabel('Number of Reviewers')
    plt.xlabel('Number of Reviews')
    plt.title('Number of Views per Reviewer')
    plt.savefig('hist_reviews.png')   


def plot_fresh_percentage(dataframe): 
    # show the distribution of percentage of total reviews rated fresh for each reviewer
    n = dataframe.size()
    n = n[n> 100]
    fresh_sum = dataframe.fresh.aggregate(lambda x: sum(i=='fresh' for i in x))
    fresh_total = dataframe.fresh.aggregate(lambda x: sum(True for i in x))
    fresh_percent = [float(x)/float(y) for x,y in zip(fresh_sum, fresh_total)]
    plt.hist(fresh_percent, 20)
    remove_border(top=False, right=False)
    plt.title('Distribution of Average Freshness Rating Per Critic')
    plt.ylabel('Number of Critics')
    plt.xlabel('Average Freshness Rating')
    plt.savefig('hist_fresh_percent.png')
 

def plot_rating_by_year(dataframe):
    # plot Top Critics Rating as a function of year
    # overplot the average for each year
    year = []
    rating = []
    # add movies who's top critics rating is greater than 0
    for n in range(len(dataframe['rtTopCriticsRating'])):  
        try:
            if dataframe['rtTopCriticsRating'][n] is not '0':
                rating.append(float(movies['rtTopCriticsRating'][n]))
                year.append(int(movies['year'][n]))
        except:
            continue

    df_new = pd.DataFrame({'year':year, 'rating': rating})        
    grouped = df_new.groupby('year')
    avg = grouped.rating.mean()
    years = grouped.year.tolist()
    years = [x[0] for x in years]
    fig = plt.figure()
    plt.scatter(year, rating, alpha = 0.3)
    plt.plot(years, avg, 'red')
    plt.title('Top Critics Ratings by Year')
    plt.xlabel('Year')
    plt.ylabel('Rating')
    axis = fig.gca()
    remove_border(axis, top=False, right=False)
    plt.savefig('ratings_by_year.png')


def make_xy(critics, vectorizer=None):
    """
    Function
    --------
    make_xy

    Build a bag-of-words training set for the review data

    Parameters
    -----------
    critics : Pandas DataFrame
        The review data from above
        
    vectorizer : CountVectorizer object (optional)
        A CountVectorizer object to use. If None,
        then create and fit a new CountVectorizer.
        Otherwise, re-fit the provided CountVectorizer
        using the critics data
        
    Returns
    -------
    X : numpy array (dims: nreview, nwords)
        Bag-of-words representation for each review.
    Y : numpy array (dims: nreview)
        1/0 array. 1 = fresh review, 0 = rotten review
    """

    vectorizer = CountVectorizer(min_df=0)
    
    # call `fit` to build the vocabulary
    quotes = critics.quote.tolist()

    vectorizer.fit(quotes)
    
    # call `transform` to convert text to a bag of words
    X = vectorizer.transform(quotes)
    Y = [int(i=='fresh') for i in critics.fresh]
    
    X = X.toarray()

    return X,Y


def calibration_plot(clf, X, Y):
    """
    Function
    --------
    calibration_plot

    Builds a plot like the one above, from a classifier and review data

    Inputs
    -------
    clf : Classifier object
        A MultinomialNB classifier
    X : (Nexample, Nfeature) array
        The bag-of-words data
    Y : (Nexample) integer array
        1 if a review is Fresh
    """  
    prediction = clf.predict_proba(X)
    
    df = pd.DataFrame(prediction, columns = ['rot_prob', 'fresh_prob'])
    df['freshness'] = Y

    bins = np.linspace(df.fresh_prob.min(), df.fresh_prob.max(), 20)
    
    groups = df.groupby(np.digitize(df.fresh_prob, bins))

    
    avg_fresh_prob = groups.fresh_prob.mean()
    
    a = groups.freshness.sum()
    a = a.tolist()
    
    size_group = groups.size()
    size_group = size_group.tolist()

    fresh_fraction = [float(x)/float(y) for x,y in zip(a,size_group)]
    
    uncertainty = [(x*(1-x)/20)**.5 for x in fresh_fraction]

    fig1 = plt.figure()
    plt.scatter(avg_fresh_prob, fresh_fraction)
    plt.errorbar(avg_fresh_prob, fresh_fraction, uncertainty)
    axis = fig1.gca()
    remove_border(axis, top=False, right=False)
    plt.title('Expected Freshness versus Observed Freshness Fraction')
    plt.xlabel('Average Freshness Probability')
    plt.ylabel('Observed Fresh Fraction')

    # overplot line y=x
    x = np.linspace(0,1,20)
    y = x
    plt.plot(x, y)
    plt.savefig('fresh_performance.png')

    # to show how many examples per bin
    groups2 = df.groupby('fresh_prob')
    fresh_probs = groups2.fresh_prob.tolist()
    fresh_probs = [x[0] for x in fresh_probs]
    
    
    fig2 = plt.figure()
    plt.hist(fresh_probs, bins =20)
    axis2 = fig2.gca()
    remove_border(axis2, top=False)
    plt.title('Distribution of Reviews Based on Freshness Probability')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Freshness Probability')
    plt.savefig('dist_freshness.png')

    return


def log_likelihood(clf, x, y):

    """
    Function
    --------
    log_likelihood

    Compute the log likelihood of a dataset according to a bayesian classifier. 
    The Log Likelihood is defined by

    L = Sum_fresh(logP(fresh)) + Sum_rotten(logP(rotten))

    Where Sum_fresh indicates a sum over all fresh reviews, 
    and Sum_rotten indicates a sum over rotten reviews
        
    Parameters
    ----------
    clf : Bayesian classifier
    x : (nexample, nfeature) array
        The input data
    y : (nexample) integer array
        Whether each review is Fresh
    """
    
    log_prediction = clf.predict_log_proba(x)
    
    #extract probabilities from classifier prediction
    fresh_prob = [n[1] for n in log_prediction]
    rot_prob = [n[0] for n in log_prediction]
    
    Sum_fresh = sum(fresh_prob)
    Sum_rotten = sum(rot_prob)

    L = Sum_fresh + Sum_rotten
    return L


def cv_score(clf, x, y, score_func):
    """
    Uses 5-fold cross validation to estimate a score of a classifier
    
    Inputs
    ------
    clf : Classifier object
    x : Input feature vector
    y : Input class labels
    score_func : Function like log_likelihood, that takes (clf, x, y) as input,
                 and returns a score
                 
    Returns
    -------
    The average score obtained by randomly splitting (x, y) into training and 
    test sets, fitting on the training set, and evaluating score_func on the test set
    
    Examples
    cv_score(clf, x, y, log_likelihood)
    """
    result = 0
    nfold = 5
    for train, test in KFold(y.size, nfold): # split data into train/test groups, 5 times
        clf.fit(x[train], y[train]) # fit
        result += score_func(clf, x[test], y[test]) # evaluate score function on held-out data
    return result / nfold # average


def maximize_cv(cv_score, log_likelihood):
    """
    Loop over many values of alpha and min_df to determine which settings are "best" in the
    sense of maximizing the cross-validated log-likelihood.
    """

    #the grid of parameters to search over
    alphas = [0, .1, 1, 5, 10, 50]
    min_dfs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    #Find the best value for alpha and min_df, and the best classifier
    best_alpha = None
    best_min_df = None
    max_loglike = -np.inf

    for alpha in alphas:
        for min_df in min_dfs:         
            vectorizer = CountVectorizer(min_df = min_df)       
            X, Y = make_xy(critics, vectorizer)
            
            
            X_training,xtest,Y_training,ytest = train_test_split(X,Y)
            clf = MultinomialNB(alpha=alpha)
            
            # replace previous alpha and min_df values with the current values if the max_loglike produced is greater
            temp = cv_score(clf, xtest, ytest,log_likelihood)
            if temp > max_loglike:
                best_alpha = alpha
                best_min_df = min_df
                max_loglike = temp
    
    return best_alpha, best_min_df

if __name__ == '__main__':

    set_display()
    
    api_key = 'xwcad7uj5uqtze4enx4mrtkk'
    movie_id = '770672122'  # toy story 3
    url = 'http://api.rottentomatoes.com/api/public/v1.0/movies/%s/reviews.json' % movie_id

    # get parameters
    options = {'review_type': 'top_critic', 'page_limit': 20, 'page': 1, 'apikey': api_key}
    data = requests.get(url, params=options).text
    data = json.loads(data)  # load a json string into a collection of lists and dicts

    movie_txt = requests.get('https://raw.github.com/cs109/cs109_data/master/movies.dat').text
    movie_file = StringIO(movie_txt) # treat a string like a file
    movies = pd.read_csv(movie_file, delimiter='\t')

    critics = build_table(movies, 3000)
    critics.to_csv('critics.csv', index=False)
    critics = pd.read_csv('critics.csv')


    # drop rows with missing data
    critics = critics[~critics.quote.isnull()]
    critics = critics[critics.fresh != 'none']
    critics = critics[critics.quote.str.len() > 0]
    
    print 'number of reviews: ' + str(len(critics)) 
    print 'there are ' + str(len(critics.groupby('title'))) + ' movies in the dataset'
    print 'there are ' + str(len(critics.groupby('critic'))) + ' critics in the dataset'

    df =  critics.groupby('critic')
    
    # save graph of distribution of reviews per reviewer
    plot_reviews(df)

    # save graph of distribution of freshness percentage per critic
    plot_fresh_percentage(df)
    
    # plot Top Critics Rating as a function of year
    plot_rating_by_year(movies)

    # build a bag of words training set for the review data
    X, Y = make_xy(critics)

    # split data into training and test set 
    X_training,xtest,Y_training,ytest = train_test_split(X,Y)
    
    # train a Naive Bayes classifier to fit the data
    clf=MultinomialNB()
    clf.fit(X_training,Y_training)

    # Your code here. Print the accuracy on the test and training dataset
    test_score = clf.score(xtest,ytest)
    training_score =   clf.score(X_training, Y_training)
    print 'Accuracy on test data: ' + str(test_score*100) + ' %%'
    print 'Accuracy on training data: ' + str(training_score*100) + ' %%'

    # plot assessments of the model calibration
    calibration_plot(clf, xtest, ytest)

    # compute log likelihood
    log_likelihood(clf, xtest, ytest)

    alpha, min_df = maximize_cv(cv_score, log_likelihood)

    # repeat calibration with optimized alpha and min_df values
    vectorizer2 = CountVectorizer(min_df = min_df)
    quotes = critics.quote.tolist()
    vectorizer2.fit(quotes)
    X2 = vectorizer2.transform(quotes)
    Y2 = [int(i=='fresh') for i in critics.fresh]
    X2 = X2.toarray()
    X_train2,xtest2,Y_train2,ytest2 = train_test_split(X2,Y2)
    clf2 = MultinomialNB(alpha = alpha)
    clf2.fit(xtest2, ytest2)
    calibration_plot(clf2, xtest2, ytest2)


    """ 
    Use classifier and vectorizer.get_feature_names() to determine most predictive
    words of fresh or rotten reviews. Produced by finding the model's probability of
    freshness for a review if a given word appears one time. This is calculated by applying
    the model to an identity matrix that represents a given word in a review.
    """
    words = vectorizer2.get_feature_names()

    # make identity matrix with dimensions according to number of words
    num_rows = len(words)   
    eye = np.eye(num_rows)
    eye_prob = clf2.predict_proba(eye)
    eye_prob = eye_prob.tolist()
    eye_fresh = [k[1] for k in eye_prob]
    eye_rot = [k[0] for k in eye_prob]
        
    df_words= pd.DataFrame({'words': words, 'fresh_prob': eye_fresh, 'eye_rot': eye_rot})
    df_words = df_words.sort_index(by='fresh_prob')

    # top 10 fresh and rotten review predictor words
    fresh_predict = df_words.tail(10)
    rot_predict = df_words.head(10)

    print 'words most predicitve of fresh reviews: ' + str(df_words.words[-10:])
    print 'words most predicitve of rotten reviews: ' + str(df_words.words[:10])



    

