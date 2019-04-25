import numpy as np
from utils import get_processed_corpus, dirichlet_denom
import logging, pickle



class LDA:

    def __init__(self, topics=2, alpha=.1, beta=.1, seed=None):
        self.K = topics
        self.alpha_ = alpha
        self.beta_ = beta
        self.rg = np.random.RandomState(seed)
    
    def set_corpus_params(self, path='data/iitm_train.csv'):
        # Preporcess the corpus
        self.corpus_tokens = get_processed_corpus(path)
        self.D = len(self.corpus_tokens)
        print('Corpus has '+str(self.D)+' documents.')

        #get vocabulary and prepare topic matrix
        self.vocab = {}
        self.reverse_vocab = {}
        ct = 0
        for doc in self.corpus_tokens:
            for word in doc:
                if not word in self.vocab.keys():
                    self.vocab[word] = ct
                    self.reverse_vocab[ct] = word
                    ct += 1

        
        self.vocab_ = list(self.vocab)
        self.V = len(self.vocab)
        print('Corpus has '+str(self.V)+' words.')
    
    def initialize_matrices(self):
        #Assign ids to words
        self.w = []
        for doc in self.corpus_tokens:
            self.w.append([])
            for word in doc:
                self.w[-1].append(self.vocab[word])

        #initialize topic assignment
        self.z = [np.zeros(len(x), dtype=int) for x in self.corpus_tokens]

        # initialize N_k for each topic: number of times word w is generated with topic k
        self.n_kw = np.zeros((self.K,self.V))
        # initialize N_i for each document: number of times topic t is generated in document i
        self.n_dk = np.zeros((self.D,self.K))

        #Sum matrices make gibbs probability computation more efficient
        #initialize sums across words for N_k matrix
        self.n_kw_sum = np.zeros(self.K)
        #initialize sums accross documents for N_i matrix
        self.n_dk_sum = np.zeros(self.K)

        #initialize alpha and beta
        self.alpha = self.alpha_*np.ones(self.K) if type(self.alpha_) == float else np.array(self.alpha_, dtype=float)
        self.beta = self.beta_*np.ones(self.V) if type(self.beta_) == float else np.array(self.beta_, dtype = float)

        assert len(self.alpha) == self.K, 'Illegal shape for alpha'
        assert len(self.beta) == self.V, 'Illegal shape for beta'

        # Parameters theta, phi
        self.theta = np.zeros((self.D, self.K))
        self.phi = np.zeros((self.K, self.V))

        #Samples for theta, phi:
        self.thetas = []
        self.phis = []
    
    '''
    Assign topics to words at random
    '''
    def initialize_topics(self):
        for i in range(self.D):
            for j in range(len(self.w[i])):
                topic = self.z[i][j] = self.rg.randint(self.K)
                word = self.w[i][j]
                #update counts
                self.n_kw[topic,word] += 1
                self.n_kw_sum[topic] += 1
                self.n_dk[i,topic] += 1
                self.n_dk_sum[topic] += 1
    
    '''
    Sample for word in document 'doc' at position 'word'
    '''
    def one_step_sample(self, doc, word):
        assert doc < self.D, 'Wrong document index'
        assert word < len(self.w[doc]), 'Wrond word index'

        prev_topic = self.z[doc][word]
        word_idx = self.w[doc][word]


        #Disregard word at (doc,word)
        self.n_kw[prev_topic,word_idx] -= 1
        self.n_kw_sum[prev_topic] -= 1
        self.n_dk[doc, prev_topic] -= 1
        self.n_dk_sum[prev_topic] -= 1

        # Evaluate topic probability p(z[i,j]|z/z[i,j],d)
        topic_probab = ((self.n_kw[:,word_idx]+ self.beta[word_idx])/(self.n_kw_sum+ np.sum(self.beta)))* \
            (self.n_dk[doc] + self.alpha)
                                

        # Normalize
        topic_probab /= np.sum(topic_probab)
        
        #Pick Topic
        new_topic = self.rg.choice(self.K, p=topic_probab)

        #Update counts
        self.n_kw[new_topic,word_idx] += 1
        self.n_kw_sum[new_topic] += 1
        self.n_dk[doc, new_topic] += 1
        self.n_dk_sum[new_topic] += 1

        #Assign
        self.z[doc][word] = new_topic

        return new_topic
    
    '''
    Do Collapsed Gibbs sampling for n iterations, with burning for 'burn' iterations
    '''
    def gibbs_train(self, niters=10, burn=1):
        print('Doing Gibbs sampling train with '+str(niters)+' steps')
        for n in range(niters):
            if n%100==0: print("Done "+str(n)+' doc_log_p='+str(self.get_document_logproaba()))
            for i in range(self.D):
                for j in range(len(self.w[i])):
                    self.one_step_sample(i,j)
            if n>burn:
                self.eval_phi_theta()
                self.thetas.append(self.theta.copy())
                self.phis.append(self.phi.copy())
                if n%100==0:
                    self.phi = np.mean(self.phis[-100:],0)
                    self.theta = np.mean(self.thetas[-100:],0)
                    self.word_ranks = (-self.phi).argsort()
                    self.print_topics()
    
    '''
    Update theta and phi
    '''
    def eval_phi_theta(self):
        self.theta = self.n_dk + self.alpha
        self.theta = (self.theta.T/np.sum(self.theta,1)).T

        self.phi = self.n_kw + self.beta
        self.phi = (self.phi.T/np.sum(self.phi,1)).T
       
        return self.theta, self.phi

    '''
    Training routine
    '''
    def fit(self, corpus_path = 'data/iitm_train_tiny.csv', n_iters=1000, burn=100):
        self.set_corpus_params(corpus_path)
        self.initialize_matrices()
        self.initialize_topics()

        self.gibbs_train(n_iters, burn)

        #Get average of sampled
        self.phi = np.mean(self.phis[-100:],0)
        self.theta = np.mean(self.thetas[-100:],0)

        self.word_ranks = (-self.phi).argsort()
        print('Done training!')

    '''
    Print top 'print_first' words for each topic along with their probabilities
    '''
    def print_topic(self,t, print_first=10):
        if print_first is None: print_first = self.V
                
        print("Topic",t,":",end=' ')
        for i in range(print_first):
            print("%f*%s "%(self.phi[t][self.word_ranks[t,i]],self.reverse_vocab[self.word_ranks[t,i]]), end=' ')
            if i<print_first-1: print("+", end=' ')
    

    def print_topics(self, print_first=10):
        print("Topics Found:")
        for t in range(self.K):
            self.print_topic(t,print_first)
            print()
    
    '''
    Get log(p(d|\alpha,\beta))
    '''
    def get_document_logproaba(self, topic_distribution=None, doc_topic_distribution=None):
        if topic_distribution is None:
            topic_distribution = self.n_kw
        if doc_topic_distribution is None:
            doc_topic_distribution = self.n_dk
        
        ans = 0
        for i in range(self.K):
            ans += (dirichlet_denom(topic_distribution[i]+self.beta))
        ans -= (self.K*dirichlet_denom(self.beta))
        for i in range(self.D):
            ans += (dirichlet_denom(doc_topic_distribution[i]+self.alpha))
        ans -= (self.D*dirichlet_denom(self.alpha))
        
        return ans
    
    '''
    Preprocessing and initialization steps for test corpus
    '''
    def init_eval(self,path = 'data/iitm_test_tiny.csv'):
        # Preprocess test corpus
        self.eval_corpus_tokens = get_processed_corpus(path)
        self.eval_D = len(self.eval_corpus_tokens)
        print('Eval Corpus has '+str(self.eval_D)+' documents.')

        self.eval_w = []
        #Word matrix to map word to index -1 if not present
        for doc in self.eval_corpus_tokens:
            self.eval_w.append([])
            for word in doc:
                if word in self.vocab:
                    self.eval_w[-1].append(self.vocab[word])
                else: self.eval_w[-1].append(-1)
        
        #Topic assignment matrix
        self.eval_z = [np.zeros(len(x), dtype=int) for x in self.eval_corpus_tokens]

        #Document topic count matrix
        self.eval_n_dk = np.zeros((self.eval_D, self.K))

        self.eval_thetas = []

        #Random assignment of topics
        for i in range(self.eval_D):
            for j in range(len(self.eval_w[i])):
                if self.eval_w[i][j] == -1:
                    self.eval_z[i][j] = -1
                else:
                    topic = self.eval_z[i][j] = self.rg.randint(self.K)
                    self.eval_n_dk[i,topic] += 1
    
    '''
    Gibbs sampling to get theta for test corpus
    '''
    def eval_one_step_sample(self, doc, word):
        assert doc < self.eval_D, 'Wrong document index'
        assert word < len(self.eval_w[doc]), 'Wrond word index'
        #If word not in vocab, ignore it
        if self.eval_w[doc][word] == -1: return -1

        prev_topic = self.eval_z[doc][word]

        self.eval_n_dk[doc, prev_topic] -= 1

        # Topic probab calculation
        topic_probab = self.phi[:,self.eval_w[doc][word]]*(self.eval_n_dk[doc]+self.alpha)

        topic_probab /= np.sum(topic_probab)

        #Choose topic
        new_topic = self.rg.choice(self.K, p=topic_probab)

        self.eval_n_dk[doc, new_topic] += 1

        self.eval_z[doc][word] = new_topic

        return new_topic
    
    '''
    Do Collapsed Gibbs sampling for n iterations, with burning for 'burn' iterations
    '''
    def eval_gibbs_train(self, niters=10, burn=1):
        for n in range(niters):
            if n%100==0: print("Done "+str(n))
            for i in range(self.eval_D):
                for j in range(len(self.eval_w[i])):
                    if self.eval_w[i][j] != -1:
                        self.eval_one_step_sample(i,j)
            if n>burn:
                self.eval_theta = self.eval_n_dk + self.alpha
                self.eval_theta = (self.eval_theta.T/np.sum(self.eval_theta,1)).T
                self.eval_thetas.append(self.eval_theta.copy())
    
    '''
    Testing routine
    '''
    def predict(self, path='data/iitm_test.csv', n_iters=1000, burn=100):
        self.init_eval(path)
        self.eval_gibbs_train(n_iters, burn)

        #Get averaged theta to predict scores
        self.eval_theta = np.mean(self.eval_thetas,0)

    def save_model(self, path):
        ds = {'phi': self.phi,
                'theta': self.theta,
                'alpha': self.alpha,
                'beta':self.beta,
                'vocab': self.vocab}
        with open(path,'wb') as fl:
            pickle.dump(ds, fl)
        
        print("Saved model at:", path)
    
    def load_model(self, path):
        with open(path, 'rb') as fl:
            ds = pickle.load(fl)
        self.phi = ds['phi']
        self.theta = ds['theta']
        self.beta = ds['beta']
        self.alpha = ds['alpha']
        self.vocab = ds['vocab']
    
    def print_eval_results(self):
        for i in range(self.eval_D):
            print('Scores for Document',i)
            order = (-self.eval_theta[i]).argsort()
            for j in range(self.K):
                print('Topic',order[j], 'Score:', self.eval_theta[i,order[j]], end=' ')
                print(self.print_topic(order[j]))
                print()


            