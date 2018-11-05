import numpy
import tensorflow as tf
from Helpers import read_tfrecords, write_tfrecords_from_memory


if __name__ == '__main__':
    # 'from_memory' or 'from_filename_list'
    data_source = 'from_memory'

    train_file = './Data/train.tfrecords'
    test_file = './Data/test.tfrecords'


    if data_source == 'from_memory':
        train_data = numpy.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=numpy.float32)
        train_label = numpy.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=numpy.float32)
        write_tfrecords_from_memory(file_name=train_file, samples=(train_data, train_label))

        test_data = numpy.array([[100, 100], [200, 200], [300, 300], [400, 400]], dtype=numpy.float32)
        test_label = numpy.array([[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=numpy.float32)
        write_tfrecords_from_memory(file_name=test_file, samples=(test_data, test_label))
    else:
        #TODO
        pass


    train_data, train_label = read_tfrecords(train_file, batch_size=1, num_epochs=2)
    test_data, test_label = read_tfrecords(test_file, batch_size=1, num_epochs=50)

    # we use try/except here to catch the exception when runing out of data in one iterator, this iterator must be train iterator!
    # so num_epochs for test iterator does not matter, BUT it has to be greater than the one of train iterator
    # we do not want to run out of test data during training.
    i = 0
    with tf.Session() as sess:
        while True:
            try:
                curr_train_data, curr_train_label = sess.run([train_data, train_label])
                print('\n\n#### %sth iteration...' %(i+1))
                print('train data:\n', curr_train_data)
                #print('train label:\n', curr_train_label)
                i = i + 1
                if i%2==0 and i!=0:
                    curr_test_data, curr_test_label = sess.run([test_data, test_label])
                    print('test data:\n', curr_test_data)
                    #print('test label:\n', curr_test_label)
            except:
                print('\ndone!')
                break
        print('Out!')
