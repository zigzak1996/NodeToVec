This is a student project for Big Data course.


RUN:

spark-submit --driver-memory 2g --class NodeToVec nodetovec_2.11-0.1.jar arg0 arg1


arg0 = "/big_data_hw_2/data/train_epin.csv" // path to input train file

arg1 = "/big_data_hw_2/data/test_epin.csv" // path to input test file

