import time
import tensorflow as tf

########### Convolutional neural network class ############
class ConvNet(object):
    def __init__(self, mode):
        self.mode = mode

    # Read train, valid and test data.
    def read_data(self, train_set, test_set):
        # Load train set.
        trainX = train_set.images
        trainY = train_set.labels
        
        
        
        # Load test set.
        testX = test_set.images
        testY = test_set.labels

        return trainX, trainY, testX, testY

    # Baseline model. step 1
    def model_1(self, X, hidden_size):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        input_layer = tf.reshape(X, [-1, 26*26*40])
        fcl = tf.layers.dense(inputs=input_layer, units=100, activation=tf.sigmoid)
        return fcl

    # Use two convolutional layers.
    def model_2(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        input_layer = tf.reshape(X, [-1, 28, 28, 1])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=40 ,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
        
        print("pool1", pool1.shape)
        
        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=40 ,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
        
        print("pool2",pool2.shape)

        fcl = pool2
        return  self.model_1(fcl, hidden_size)


    # Replace sigmoid with ReLU.
    def model_3(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        input_layer = ConvNet.model_2(self, X, hidden_size)

        fcl = tf.layers.dense(inputs=input_layer, units=100, activation=tf.nn.relu)

        return  fcl


    # Add one extra fully connected layer.
    def model_4(self, X, hidden_size, decay):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        input_layer = ConvNet.model_3(self, X, hidden_size)

        fcl = tf.layers.dense(inputs=input_layer, units=100, activation=tf.nn.relu)

        return  fcl


    # Use Dropout now.
    def model_5(self, X, hidden_size, is_train, decay):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        input_layer = ConvNet.model_4(self, X, hidden_size, decay)

        dropout = tf.layers.dropout(inputs=input_layer, rate=0.5, training=is_train)

        fcl = dropout
        return  fcl

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS, train_set, test_set):
        class_num = 10
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate
        hidden_size = FLAGS.hiddenSize
        decay = FLAGS.decay

        trainX, trainY, testX, testY = self.read_data(train_set, test_set)

        input_size = trainX.shape[1]
        train_size = trainX.shape[0]
        test_size = testX.shape[0]
        
        trainX = trainX.reshape((-1, 28, 28, 1))
        testX = testX.reshape((-1, 28, 28, 1))       
        
        # print("Test images = ", testX.shape, "Test labels = ", testY.shape, "Train images = ", trainX.shape, "Train labels = ", trainY.shape)
        # print("Inp size", input_size, "Train size", train_size, "Test size", test_size)

        with tf.Graph().as_default():
            # Input data
            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            Y = tf.placeholder(tf.int32, [None])
            is_train = tf.placeholder(tf.bool)

            # model 1: base line
            if self.mode == 1:
                features = self.model_1(X, hidden_size)

            # model 2: use two convolutional layer
            elif self.mode == 2:
                features = self.model_2(X, hidden_size)

            # model 3: replace sigmoid with relu
            elif self.mode == 3:
                features = self.model_3(X, hidden_size)

            # model 4: add one extral fully connected layer
            elif self.mode == 4:
                features = self.model_4(X, hidden_size, decay)

            # model 5: utilize dropout
            elif self.mode == 5:
                features = self.model_5(X, hidden_size, is_train, decay)

            # ======================================================================
            # Define softmax layer, use the features.
            # ----------------- YOUR CODE HERE ----------------------
            #            
            
            print("Feat", features.shape)
            
            logits = tf.layers.dense(inputs=features, units=10)

            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": tf.argmax(input=logits, axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

            # ======================================================================
            # Define loss function, use the logits.
            # ----------------- YOUR CODE HERE ----------------------

            print("Logit Shape", logits.shape)
            
            loss = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=logits)
            
            # print(loss.shape)

            # ======================================================================
            # Define training op, use the loss.
            # ----------------- YOUR CODE HERE ----------------------
            #
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
                
                
            # ======================================================================
            # Define accuracy op.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # eval_metric_ops = {
            #     "accuracy": tf.metrics.accuracy(labels=Y, predictions=predictions["classes"])}
            #     
            # accuracy = tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL, loss=loss, predictions=predictions["classes"], train_op=train_op, eval_metric_ops=eval_metric_ops)
            # 
            # accuracy.__name__ = "foo"
            
            accuracy_ten, accuracy = tf.metrics.accuracy(labels=Y, predictions=predictions["classes"])

            # ======================================================================
            # Allocate percentage of GPU memory to the session.
            # If you system does not have GPU, set has_GPU = False
            #
            has_GPU = True
            if has_GPU:
                gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
                config = tf.ConfigProto(gpu_options=gpu_option)
            else:
                config = tf.ConfigProto()
            
            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()

            # Create TensorFlow session with GPU setting.
            with tf.Session(config=config) as sess:
                tf.global_variables_initializer().run()
                sess.run(init_g)
                sess.run(init_l)

                for i in range(num_epochs):
                    print(20 * '*', 'epoch', i + 1, 20 * '*')
                    
                    start_time = time.time()
                    s = 0
                    while s < train_size:
                        e = min(s + batch_size, train_size)
                        batch_x = trainX[s: e]
                        batch_y = trainY[s: e]
                        print(batch_x.shape, batch_y.shape, trainX.shape, trainY.shape, e,  s)
                        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
                        s = e
                    end_time = time.time()
                    print ('the training took: %d(s)' % (end_time - start_time))


                    total_correct = sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False})
                    print ('accuracy of the trained model %f' % (total_correct / testX.shape[0]))
                    print ()

                return sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False}) / testX.shape[0]




