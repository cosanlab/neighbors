def create_sub_by_item_matrix(df):
    ''' Convert a pandas long data frame of a single rating into a subject by item matrix
    
        Args:
            df: pandas dataframe instance.  Must have column names ['Subject','Item','Rating]
            
    '''
    if not isinstance(df,pd.DataFrame):
        raise ValueError('df must be pandas instance')
    if np.any([not x in df.columns for x in ['Subject','Item','Rating']]):
        raise ValueError("df must contain ['Subject','Item','Rating] as column names")
        
    ratings = pd.DataFrame(columns=df.Item.unique(),index=df['Subject'].unique())
    for row in df.iterrows():
        ratings.loc[row[1]['Subject'], row[1]['Item']] = float(row[1]['Rating'])
    return ratings.astype(float)

def train_test_split(ratings, n_train_items):
    ''' Split ratings matrix into train and test items
    
    Args:
        ratings: pandas rating object instance
        train_items: number of items for test dictionary or list of specific items
        
    Returns:
        test: dictionary of test items
        train: dictionary of train items
        
    '''
    
    if not isinstance(ratings,pd.DataFrame):
        raise ValueError('ratings must be a pandas dataframe instance')

    train = ratings.copy()
    train.loc[:,:] = np.full(ratings.shape, np.nan)
    test = ratings.copy()
    
    for sub in ratings.index:
        sub_train_rating_item =  np.random.choice(ratings.columns,replace=False, size=n_train_items)
        train.loc[sub, sub_train_rating_item] = ratings.loc[sub, sub_train_rating_item]
        test.loc[sub, sub_train_rating_item] = np.nan
    assert(~np.any(train==test))
    return train,test

def similarity(ratings, metric='correlation'):
    ''' Calculate Subject similarity across items
    
        Args:
            ratings: pandas dataframe instance of ratings
            metric: type of similarity {"correlation","cosine"}
        Returns:
            similarity matrix
    '''
    
    if metric is 'correlation':
        sim = pd.DataFrame(np.zeros((ratings.shape[0],ratings.shape[0])))
        sim.columns=ratings.index
        sim.index=ratings.index
        for x in ratings.iterrows():
            for y in ratings.iterrows():
                sim.loc[x[0],y[0]] = pearsonr(x[1][(~x[1].isnull()) & (~y[1].isnull())],y[1][(~x[1].isnull()) & (~y[1].isnull())])[0] 
        return sim
    elif metric is 'cosine':
        sim = ratings.dot(ratings.T)
        norms = np.array([np.sqrt(np.diagonal(sim.values))])
        return (sim.values / norms / norms.T)

def predict(ratings, sim, n_subjects=None):
    ''' Predict Subject's missing items using similarity based collaborative filtering.
    
        Args:
            ratings: pandas dataframe instance of ratings
            n_subjects: number of closest neighbors to use
        Returns:
            predicted rating: (pd.DataFrame instance)
    '''
    pred = pd.DataFrame(np.zeros(ratings.shape))
    pred.columns = ratings.columns
    pred.index = ratings.index
    for row in ratings.iterrows():
        if n_subjects is not None:
            top_subjects = sim.loc[row[0]].drop(row[0]).sort_values(ascending=False)[0:n_subjects]
        else:
            top_subjects = sim.loc[row[0]].drop(row[0]).sort_values(ascending=False)
        for col in ratings.iteritems():
            pred.loc[row[0],col[0]] = np.dot(top_subjects,ratings.loc[top_subjects.index,col[0]].T)/len(top_subjects)
    return pred

def evaluate(pred, actual, metric='correlation'):
    ''' Evaluate overall performance of prediction
        Args:
            pred: pd.DataFrame instance of predicted ratings
            actual: pd.DataFrame instance of true ratings
            metric: type of metric to evaluate can be ["correlation","mse","accuracy"]
    '''
    if metric is 'correlation':
        actual = actual.values.flatten()
        pred = pred.values.flatten()
        return pearsonr(pred[(~np.isnan(actual)) & (~np.isnan(pred))], actual[(~np.isnan(actual)) & (~np.isnan(pred))])[0]
    elif metric is 'mse':
        actual = actual.values.flatten()
        pred = pred.values.flatten()
        return np.mean((pred[(~np.isnan(actual)) & (~np.isnan(pred))] - actual[(~np.isnan(actual)) & (~np.isnan(pred))])**2)
    elif metric is 'accuracy':
        return np.sum(pred==actual)
    else:
        raise ValueError('Metric must be ["correlation","mse","accuracy"]')
        
def nmf_multiplicative(X, n_components=None, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6, verbose=True):
    ''' Train non negative matrix factorization model using multiplicative updates.  Allows masking to only learn 
        the training weights.
    
    '''
    
    mask = ~np.isnan(X.values)
    train[train.isnull()] = 0
    X = X.values

    eps = 1e-5

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    # Initial guesses for solving X ~= WH. H is random [0,1] scaled by sqrt(X.mean() / n_components)
    avg = np.sqrt(np.nanmean(X)/n_components)
    H = avg*np.random.rand(n_features, n_components) # H = Y
    W = avg*np.random.rand(n_samples, n_components)   # W = A
    masked_X = mask * X
    X_est_prev = np.dot(W, H)

    for i in range(1, max_iter + 1):
        # Update W: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        W *= np.dot(masked_X, H.T) / np.dot(mask * np.dot(W,H), H.T)
#         W = np.maximum(W, eps)

        # Update H: Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        H *= np.dot(W.T, masked_X) / np.dot(W.T, mask * dot(W,H))
#         H = np.maximum(H, eps)

        # Evaluate
        if i % 5 == 0 or i == 1 or i == max_iter:
            X_est = np.dot(W,H)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est
            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            if verbose:
                print 'Iteration {}:'.format(i),
                print 'fit residual', np.round(fit_residual, 4),
                print 'total residual', np.round(curRes, 4)
            if curRes < error_limit or fit_residual < fit_error_limit:
                break
    return W, H

# def get_mse(pred, actual):
#     ''' Get Mean Squared Error ignoring Missing Values '''
#     actual = actual.values.flatten()
#     pred = pred.values.flatten()
#     return np.mean((pred[(~np.isnan(actual)) & (~np.isnan(pred))] - actual[(~np.isnan(actual)) & (~np.isnan(pred))])**2)


class ExplicitMF():
    def __init__(self, 
                 ratings,
                 mask=None,
                 n_factors=40,
                 learning='sgd',
                 item_fact_reg=0.0, 
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 learning_rate=0.1,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
        
        mask: (ndarray)
            Boolean matrix indicating missing values
        n_factors : (int)
            Number of latent factors to use in matrix 
            factorization model.  If None, will use full feature set
        learning : (str)
            Method of optimization. Options include 
            'sgd' or 'als'.
        
        item_fact_reg : (float)
            Regularization term for item latent factors
        
        user_fact_reg : (float)
            Regularization term for user latent factors
            
        item_bias_reg : (float)
            Regularization term for item biases
        
        user_bias_reg : (float)
            Regularization term for user biases
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        

        self.ratings = ratings
        self.mask = mask
        self.n_users, self.n_items = ratings.shape
        if n_factors is not None:
            self.n_factors = n_factors
        else:
            self.n_factors = self.n_items
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning = learning
        if self.learning == 'sgd':
            self.learning_rate = learning_rate

            if self.mask is not None:
                self.sample_row, self.sample_col = self.mask.nonzero()
            else:
                self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose

        self.initialize()
        
    def initialize(self):
        """ Initialize variables for matrix factorization """
        
        # initialize latent vectors        
        self.user_vecs = np.random.normal(scale=1./self.n_factors,
                                          size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.normal(scale=1./self.n_factors,
                                          size=(self.n_items, self.n_factors))
        # Initialize biases
        if self.learning == 'sgd':
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            if self.mask is not None:
                self.global_bias = np.mean(self.ratings[~mask])
            else:
                self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
                
    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in xrange(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI), 
                                             ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda
            
            for i in xrange(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI), 
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors
    
    def train(self, n_iter=10):
        """ Train model for n_iter iterations. Can be called multiple times for further training."""
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print '\tcurrent iteration: {}'.format(ctr)
            if self.learning == 'als':
                self.user_vecs = self.als_step(self.user_vecs, 
                                               self.item_vecs, 
                                               self.ratings, 
                                               self.user_fact_reg, 
                                               type='user')
                self.item_vecs = self.als_step(self.item_vecs, 
                                               self.user_vecs, 
                                               self.ratings, 
                                               self.item_fact_reg, 
                                               type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)

            e = (self.ratings[u,i] - prediction) # error
            
            # Update biases
            self.user_bias[u] += (self.learning_rate * (e - self.user_bias_reg * self.user_bias[u]))
            self.item_bias[i] += (self.learning_rate * (e - self.item_bias_reg * self.item_bias[i]))
            
            # Update latent factors
            self.user_vecs[u, :] += (self.learning_rate * (e * self.item_vecs[i, :] - 
                                     self.user_fact_reg * self.user_vecs[u,:]))
            self.item_vecs[i, :] += (self.learning_rate *
                                    (e * self.user_vecs[u, :] - self.item_fact_reg * self.item_vecs[i,:]))

    def predict(self, u, i):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            return prediction
    
    def predict_all(self):
        """ Predict ratings for every user and item."""
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in xrange(self.user_vecs.shape[0]):
            for i in xrange(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
                
        return predictions
    
    def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
        """
        Keep track of MSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).
        
        The function creates two new class attributes:
        
        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print 'Iteration: {}'.format(n_iter)
            self.train(n_iter - iter_diff)

            predictions = self.predict_all()
            if self.mask is not None:
                self.train_mse += [get_mse(predictions, self.ratings, mask=self.mask)]
                self.test_mse += [get_mse(predictions, test, mask=~self.mask)]                
            else:
                self.train_mse += [get_mse(predictions, self.ratings)]
                self.test_mse += [get_mse(predictions, test)]
            
            if self._v:
                print 'Train mse: ' + str(self.train_mse[-1])
                print 'Test mse: ' + str(self.test_mse[-1])
            iter_diff = n_iter
            
def get_mse(pred, actual, mask=None):
    ''' Get Mean Squared Error ignoring Missing Values '''
    if mask is not None:
        pred = pred[mask]
        actual = actual[mask]
        return np.mean((pred-actual)**2)
    else:
        actual = actual.flatten()
        pred = pred.flatten()
        return np.mean((pred[(~np.isnan(actual)) & (~np.isnan(pred))] - actual[(~np.isnan(actual)) & (~np.isnan(pred))])**2)

def plot_learning_curve(iter_array, model):
    plt.plot(iter_array, model.train_mse,
             label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse,
             label='Test', linewidth=5)

    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel('iterations', fontsize=30);
    plt.ylabel('MSE', fontsize=30);
    plt.legend(loc='best', fontsize=20);

def plot_predictions(ratings, predicted_ratings, mask=None):
    f, ax = plt.subplots(nrows=1,ncols=3, figsize=(15,8))
    sns.heatmap(ratings,vmax=100,vmin=0,ax=ax[0],square=False)
    ax[0].set_title('Actual User/Item Ratings')
    sns.heatmap(predicted_ratings,vmax=100,vmin=0,ax=ax[1],square=False)
    ax[1].set_title('Predicted User/Item Ratings')
    if isinstance(ratings,pd.DataFrame):
        actual = test.values.flatten()
    elif isinstance(ratings,np.ndarray):
        actual = test.flatten()
    else:
        raise ValueError('Make sure ratings are pandas data frame or numpy array.')
    if isinstance(predicted_ratings,pd.DataFrame):
        pred = predicted_ratings.values.flatten()
    elif isinstance(predicted_ratings,np.ndarray):
        pred = predicted_ratings.flatten()
    else:
        raise ValueError('Make sure predicted_ratings are pandas data frame or numpy array.')
    if mask is not None:
        if isinstance(mask,pd.DataFrame):
            mask = mask.values.flatten()
        elif isinstance(mask,np.ndarray):
            mask = mask.flatten()
        else:
            raise ValueError('Make sure mask is a pandas data frame or numpy array.')
        pred = pred[~mask]
        actual = actual[~mask]
    ax[2].scatter(actual[(~np.isnan(actual)) & (~np.isnan(pred))],pred[(~np.isnan(actual)) & (~np.isnan(pred))])
    ax[2].set_xlabel('Actual Ratings')
    ax[2].set_ylabel('Predicted Ratings')
    ax[2].set_title('Predicted Ratings')
    r = pearsonr(actual[(~np.isnan(actual)) & (~np.isnan(pred))],pred[(~np.isnan(actual)) & (~np.isnan(pred))])
    print 'Correlation: %s' % r[0]
    return f,r

def subject_correlations(ratings, predicted_ratings, mask=None):
    '''Calculate observed/predicted correlation for each subject in matrix'''
    if mask is not None:
        r = []
        for i,sub in enumerate(ratings.index):
            r.append(pearsonr(ratings.loc[sub,mask[i,:]],predicted_ratings[i,mask[i,:]])[0])
    else:
        r = []
        for i,sub in enumerate(ratings.index):
            r.append(pearsonr(ratings.loc[sub,:],predicted_ratings[i,:])[0])
    return r        

rr = subject_correlations(ratings, predicted_ratings, mask=mask)