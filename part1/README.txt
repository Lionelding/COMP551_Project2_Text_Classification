To run the Naive Bayes classifier you have three options

To perform cross-validation you must change the alpha values in main(). Then provide the stdin options training_file_path_train_in training_file_path_train_out crossvalidate

To generate results given a test set, provide training_file_path_train_in training_file_path_train_out test_file_path_test_in. the results will be printed to the screen, pipe to csv.

To test given the provided results specify training_file_path_train_in training_file_path_train_out test. The split will be 80-20. The error will be printed to the screen.
