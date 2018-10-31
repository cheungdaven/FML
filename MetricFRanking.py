import tensorflow as tf
import random

from evaluation import *


class MetricFRanking():

    def __init__(self ,sess,  num_users, num_items, learning_rate = 0.1, epoch=200, N = 100, batch_size=1024 * 3 ):
        self.lr = learning_rate
        self.epochs = epoch
        self.N = N
        self.num_users = num_users
        self.num_items = num_items
        self.batch_size = batch_size
        self.clip_norm = 1
        self.sess = sess
        self.beta =1.5#2.5#0.6#2.5#1.5

    def run(self, train_data, unique_users, neg_train_matrix, test_matrix):
        self.cf_user_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_user_input')
        self.cf_item_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_item_input')
        self.y = tf.placeholder("float", [None], 'y')


        U = tf.Variable(tf.random_normal([self.num_users, self.N], stddev=1 / (self.N ** 0.5)), dtype=tf.float32)
        V = tf.Variable(tf.random_normal([self.num_items, self.N], stddev=1 / (self.N ** 0.5)), dtype=tf.float32)

        users = tf.nn.embedding_lookup(U ,self.cf_user_input)
        pos_items = tf.nn.embedding_lookup(V, self.cf_item_input)


        L = tf.Variable(tf.random_normal([self.num_users, self.N], stddev=1 / (self.N ** 0.5)), dtype=tf.float32)


        self.pos_distances = tf.reduce_sum(tf.squared_difference(users  , pos_items) ,1, name="pos_distances")   
        self.pred = tf.reduce_sum(tf.nn.dropout(tf.squared_difference(users, pos_items),0.95), 1, name="pred")

        self.loss = tf.reduce_sum((1 + 0.1* self.y)*  tf.square((self.y * self.pred + (1 - self.y)  * tf.nn.relu(self.beta * (1 - self.y) - self.pred))))
        gds = []
        self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.loss, var_list=[U, V])
        '''
        with tf.control_dependencies(gds):
            self.optimizer = gds + [[tf.assign(U, tf.clip_by_norm(U, self.clip_norm, axes=[1])),tf.assign(V, tf.clip_by_norm(V, self.clip_norm, axes=[1])),]]
        '''
        clip_U = tf.assign(U, tf.clip_by_norm(U, self.clip_norm, axes=[1]))
        clip_V = tf.assign(V, tf.clip_by_norm(V, self.clip_norm, axes=[1]))
        # initialize model
        init = tf.global_variables_initializer()


        temp = train_data.tocoo()

        item = list(temp.col.reshape(-1))
        # print(np.shape(item))
        # print(type(item))
        user = list(temp.row.reshape(-1))
        rating = list(temp.data)
        print(len(rating))
        self.sess.run(init)
        # train and test the model
        sample_size = 0



        for epoch in range(self.epochs):
            # print("epoch:"+str(epoch))
            user_temp = user[:]
            item_temp = item[:]
            rating_temp = rating[:]
            # print(len(rating))
            if epoch % 5 == 0:
                user_append = []
                item_append = []
                values_append = []
                for u in range(self.num_users):
                    if sample_size > len(neg_train_matrix[u]):
                        list_of_random_items = random.sample(neg_train_matrix[u], len(neg_train_matrix[u]))
                    else:
                        list_of_random_items = random.sample(neg_train_matrix[u], sample_size)
                    user_append += [u] * sample_size
                    item_append += list_of_random_items
                    values_append += [0] * sample_size
            item_temp += item_append
            user_temp += user_append
            rating_temp += values_append
            self.num_training = len(rating_temp)
            total_batch = int(self.num_training / self.batch_size)
            idxs = np.random.permutation(self.num_training)
            user_random = list(np.array(user_temp)[idxs])
            item_random = list(np.array(item_temp)[idxs])

            rating_random = list(np.array(rating_temp)[idxs])

            for i in range(total_batch):
                batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
                batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
                batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

                _, c, p, _, _ = self.sess.run((self.optimizer, self.loss, self.pos_distances, clip_U, clip_V), feed_dict={self.cf_user_input: batch_user,
                                                                             self.cf_item_input: batch_item,
                                                                             self.y: batch_rating})
                avg_cost = c

                if i % 1000 == 0:
                    print("Index: %04d; Epoch: %04d; loss = %.9f" % (i + 1, epoch, np.mean(avg_cost)))

            if (epoch) % 2 == 0 and epoch >= 0:  # 300

                pred_ratings_10 = {}
                pred_ratings_5 = {}
                pred_ratings = {}
                ranked_list = {}
                count = 0
                p_at_5 = []
                p_at_10 = []
                r_at_5 = []
                r_at_10 = []
                map = []
                mrr = []
                ndcg = []
                for u in unique_users:
                    user_ids = []
                    count += 1
                    user_neg_items = neg_train_matrix[u]
                    item_ids = []

                    for j in user_neg_items:
                        item_ids.append(j)
                        user_ids.append(u)
                    ratings = - self.sess.run([self.pos_distances]
                                             ,feed_dict={self.cf_user_input: user_ids, self.cf_item_input: item_ids})[0]
                    # print(ratings)
                    neg_item_index = list(zip(item_ids, ratings))

                    ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
                    pred_ratings[u] = [r[0] for r in ranked_list[u]]
                    pred_ratings_5[u] = pred_ratings[u][:5]
                    pred_ratings_10[u] = pred_ratings[u][:10]

                    p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], test_matrix[u])
                    p_at_5.append(p_5)
                    r_at_5.append(r_5)
                    p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], test_matrix[u])
                    p_at_10.append(p_10)
                    r_at_10.append(r_10)
                    map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], test_matrix[u])
                    map.append(map_u)
                    mrr.append(mrr_u)
                    ndcg.append(ndcg_u)

                print("-------------------------------")
                print("precision@10:" + str(np.mean(p_at_10)))
                print("recall@10:" + str(np.mean(r_at_10)))
                print("precision@5:" + str(np.mean(p_at_5)))
                print("recall@5:" + str(np.mean(r_at_5)))
                print("map:" + str(np.mean(map)))
                print("mrr:" + str(np.mean(mrr)))
                print("ndcg:" + str(np.mean(ndcg)))


from loaddata import *

if __name__ == '__main__':
    train_data, neg_train_matrix, test_data, test_matrix, num_users, num_items, unique_users \
        = load_ranking_data(path="data/filmtrust_ratings.dat",header=['user_id', 'item_id', 'rating'], sep=" ")
    with tf.Session() as sess:
        model = MetricFRanking(sess, num_users, num_items)
        model.run(train_data, unique_users, neg_train_matrix, test_matrix)
