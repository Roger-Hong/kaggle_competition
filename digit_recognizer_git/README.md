#Digit Recognizer

Competition homepage: [https://www.kaggle.com/c/digit-recognizer](https://www.kaggle.com/c/digit-recognizer)

###Result:

1. Baseline CNN 2 layer
	* Layer 1: 32 node; Layer 2: 64 node; Drop rate: 0.5
	* Train steps: 200 rounds of all train images
	* **Kaggle accuracy**: 0.99142 (Rank: 1028)

###Mise:

1. Loading training data:
	* The easiest way to read/write data to Google cloud storage (gs://...) is `from tensorflow.python.lib.io import file_io`.
	* And it is not reasonable and feasible to load the whole dataset into memory when training.
	* The reasonable approach to handle training data in csv file is to read few lines of code (E.g. 1000 lines) as chunk. `pandas.read_csv()` returns an iterator to load a chunk of data.
	* Example of loading data:
```
	file_path = os.path.join(FLAGS.dataset_path, filename)
	file_stream = file_io.FileIO(file_path, mode="r")
	iterator = pd.read_csv(StringIO(file_stream.read()), sep=',', iterator=True)
	for chunk_idx in range(...):
		chunk_data = iterator.get_chunk(CHUNK_SIZE).values
```

2. Code testing and training:
	* test to run the code locally with a small dataset or only one iteration of train data set before deploy to gcloud ml engine.
	* Add proper amount of log for monitoring the progress of training on gcloud.
