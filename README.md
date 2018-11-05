# TF-read-write-data

Tensorflow has 4 general ways to read data, link: https://www.tensorflow.org/api_guides/python/reading_data:
1. tf.data
2. Feeding
3. QueueRunner
4. Preloaded data

Trying `tf.data`...
1. `tf.python_io.TFRecordWriter`: the writer
2. `TFRecordDataset`: high level API, it has two types of iterators:

   a. `make_initializable_iterator`
   
   b. `make_one_shot_iterator`
   
3. implemented a simple example here which applies two iterators: `one for train data, the other one for test data`. In this way we can switch between them during training.

    a. pay attention to the `reminder` (when num_samples/batch_size is not an interger)
    
    b. we need to make sure we `alawys have test data before finishing training`, which means the `test iterator` should not stop before `train iterator`

![Test](https://github.com/kevin28520/TF-read-data/blob/master/test.PNG?raw=true)

#### TODO

write tfrecords from seperated samples such as images stored on hard drive.
