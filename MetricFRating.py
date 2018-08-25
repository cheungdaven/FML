import tensorflow as tf
from evaluation import *

class MetricFRating():

    def __init__(self ,sess,  num_users, num_items, learning_rate = 0.05, epoch=200, N = 150, batch_size=256):
        self.lr = learning_rate
        self.epochs = epoch
        self.N = N
        self.num_users = num_users
        self.num_items = num_items
        self.batch_size = batch_size
        self.sess = sess
        self.max_rating = 5
        self.min_rating = 0
        self.clip_norm = 1

    def run(self, train_data, test_data):
        self.cf_user_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_user_input')
        self.cf_item_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_item_input')
        self.y = tf.placeholder("float", [None], 'y')


        U = tf.Variable(tf.random_normal([self.num_users, self.N], mean=0.08, stddev=0.03), dtype=tf.float32)
        V = tf.Variable(tf.random_normal([self.num_items, self.N], mean=0.08, stddev=0.03), dtype=tf.float32)
        B_u = tf.Variable(tf.random_normal([self.num_users],  stddev=0.001))
        B_v = tf.Variable(tf.random_normal([self.num_items],  stddev=0.001))

        bias_u = tf.nn.embedding_lookup(B_u ,self.cf_user_input)
        bias_v = tf.nn.embedding_lookup(B_v ,self.cf_item_input)
        users = tf.nn.embedding_lookup(U ,self.cf_user_input)
        pos_items = tf.nn.embedding_lookup(V, self.cf_item_input)

        temp = train_data.tocoo()
        item = list(temp.col.reshape(-1))
        user = list(temp.row.reshape(-1))
        rating = list(temp.data)
        mu = np.mean(rating)

        self.pos_distances = tf.clip_by_value( tf.reduce_sum( tf.square(users - pos_items)  ,1) + bias_u + bias_v +   (self.max_rating - mu),self.min_rating, self.max_rating)
        self.pred_distances = tf.clip_by_value(tf.reduce_sum( tf.nn.dropout(tf.square(users - pos_items), 0.95) ,1)  + bias_u + bias_v +   (self.max_rating - mu)  , self.min_rating, self.max_rating)

        self.loss = tf.reduce_sum( ( 1+  0.2 * tf.abs(self.y - (self.max_rating )/2)) * tf.square(  (self.max_rating - self.y )  - self.pred_distances)  ) + 0.01* (tf.norm(B_u) + tf.norm(B_v) )
        self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
        clip_U = tf.assign(U, tf.clip_by_norm(U, self.clip_norm, axes=[1]))
        clip_V = tf.assign(V, tf.clip_by_norm(V, self.clip_norm, axes=[1]))

        init = tf.global_variables_initializer()
        self.sess.run(init)
        # train and test the model
        for epoch in range(self.epochs):

            self.num_training = len(rating)
            total_batch = int(self.num_training / self.batch_size)
            idxs = np.random.permutation(self.num_training)

            user_random = list(np.array(user)[idxs])
            item_random = list(np.array(item)[idxs])

            rating_random = list(np.array(rating)[idxs])

            for i in range(total_batch):
                batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
                batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
                batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

                _, c, _, _ = self.sess.run((self.optimizer, self.loss, clip_U, clip_V), feed_dict={self.cf_user_input: batch_user,
                                                                             self.cf_item_input: batch_item,
                                                                             self.y: batch_rating})
                avg_cost = c

                if i % 1000 == 0:
                    print("Index: %04d; Epoch: %04d; loss = %.9f" % (i + 1, epoch, np.mean(avg_cost)))


            if (epoch) % 1 == 0 :

                error = 0
                error_mae = 0
                test_set = list(test_data.keys())
                for (u, i) in test_set:
                    pred_rating_test = self.max_rating - self.sess.run([self.pos_distances]
                                             ,feed_dict={self.cf_user_input: [u], self.cf_item_input: [i]})[0]
                    if pred_rating_test < 0:
                        pred_rating_test = 0
                    elif pred_rating_test > self.max_rating:
                        pred_rating_test = self.max_rating

                    error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
                    error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))

                print("RMSE:" + str(np.sqrt(error / len(test_set))[0]) + "; MAE:"+str((error_mae / len(test_set)) [0]) )


from loaddata import *

if __name__ == '__main__':
    train_data, test_data, n_user, n_item, neg_user_item_matrix, train_user_item_matrix, unqiue = load_rating_data(path="data/u.data", test_size=0.1, sep="\t")
    with tf.Session() as sess:
        model = MetricFRating(sess, n_user, n_item)
        model.run(train_data, test_data)