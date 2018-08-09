from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf
import SimpleHTTPServer
import SocketServer

# tf.logging.set_verbosity(tf.logging.INFO)

# COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
#            "dis", "tax", "ptratio", "medv"]
# FEATURES = ["crim", "zn", "indus", "nox", "rm",
#             "age", "dis", "tax", "ptratio"]
# LABEL = "medv"



# def get_input_fn(data_set, num_epochs=None, shuffle=True):
#   return tf.estimator.inputs.pandas_input_fn(
#       x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
#       y=pd.Series(data_set[LABEL].values),
#       num_epochs=num_epochs,
#       shuffle=shuffle)



# training_set = pd.read_csv("boston_train.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
# test_set = pd.read_csv("boston_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)


# prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

# feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

# regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=[10, 10], model_dir="/tmp/boston_model")

# regressor.train(input_fn=get_input_fn(training_set), steps=5000)

# ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
# loss_score = ev["loss"]
# print("Loss: {0:f}".format(loss_score))

# y = regressor.predict(input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
# predictions = list(p["predictions"] for p in itertools.islice(y, 6))
# print("Predictions: {}".format(str(predictions)))




def main():
      
  PORT = 8000

  Handler = SimpleHTTPServer.SimpleHTTPRequestHandler

  httpd = SocketServer.TCPServer(("", PORT), Handler)

  print "serving at port", PORT
  httpd.serve_forever()

if __name__== "__main__":
  main()