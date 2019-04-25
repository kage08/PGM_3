from lda import LDA

train_corpus = 'data/worldnews_train.csv'
test_corpus = 'data/worldnews_test.csv'

alpha = 0.01
beta = 0.01
topics = 5


model = LDA(topics,alpha, beta)
model.fit(train_corpus, n_iters=10000, burn=8000)

model.print_topics()

x = input('Press key to start evaluation')

model.predict(test_corpus, n_iters=1000, burn=300)

model.print_eval_results()