import tensorflow as tf


def byte_feature(value):
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return feature

def write_tfrecords_from_memory(file_name, samples):
    '''
    writing tfrecords from memory, if the dataset is not huge, we can just read all of them into memory
    '''
    data = samples[0]
    label = samples[1]
    n_sample = len(label)
    with tf.python_io.TFRecordWriter(file_name) as writer:
        for i in range(n_sample):
            single_data = data[i].tostring()
            single_label = label[i].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'data' : byte_feature(single_data),
                        'label' : byte_feature(single_label)
                    }))
            writer.write(example.SerializeToString())


# def write_tfrecords_from_filename_list(file_name, data_list):
#     '''
#     if the dataset is too big, we are going to build a First In First Out queue to read and write the data
#     '''
#     n_samples = len(data_list)
#     with tf.python_io.TFRecordWriter(file_name) as writer:
#         for i in range(n_samples):
#             img = plt.imread(data_list[i]).tostring()
#             label = numpy.array(0.0) if 'cat' in data_list[i] else numpy.array(1.0)
#             label = label.tostring()
#
#             example = tf.train.Example(
#                 features = tf.train.Features(
#                     feature={
#                         'data': byte_feature(img),
#                         'label': byte_feature(label)
#                     }))
#             writer.write(example.SerializeToString())


def decode(serialized_example):
    example = tf.parse_single_example(serialized_example,
                                      features={
                                          'data': tf.FixedLenFeature([], tf.string),
                                          'label': tf.FixedLenFeature([], tf.string),
                                      })
    data = tf.decode_raw(example['data'], tf.float32)
    label = tf.decode_raw(example['label'], tf.float32)
    return data, label


def read_tfrecords(file, batch_size, num_epochs):
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.map(decode)
    dataset = dataset.shuffle(1000 + 3 * batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next = iterator.get_next()
    return next
