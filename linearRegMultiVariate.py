import collections
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gdp_data = []
interest_rate_data = []
sales_data = []


def read_data():
    # the dictionary will have gdp as key and
    # format of the file ---> year,gdp,loan_interest_rate,sale
    data = open("vehicle_sale_data_multivariate", "r")
    sale_gdp_interest_rate = collections.OrderedDict()
    for line in data.readlines()[1:] :
        record = line.split(",")
        sales_data.append(float(record[1].replace('\n', "")))
        gdp_data.append(float(record[2]))
        interest_rate_data.append(float(record[3]))


# draw graph of sample data
def draw_graph():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(gdp_data, interest_rate_data, sales_data, 'ro', label='Vehicle sales data')
    ax.legend()
    plt.show()


def createModel():
    global gdp_data
    global interest_rate_data
    global sales_data

    # since we have 2 variables (gdp as x1 and auto loan interest rate as x2) the equation format will be
    # y = c + w1*x1 + w2*x2,  c is also called bias

    w1 = tf.Variable(tf.ones(1.0, 1.0), name="w1", dtype="float")  # add to graph, initialize to 1
    w2 = tf.Variable(tf.ones(1.0, 1.0), name="w2", dtype="float")  # add to graph, initialize to 1
    b = tf.Variable(tf.ones(1.0), name="b", dtype="float") # add to graph, initialize to 1

    X1 = tf.placeholder(tf.float32, [None,1], name="x1") #placeholder means its value has to be fed later and value has to be of float type
    X2 = tf.placeholder(tf.float32, [None,1], name="x2")

    # we multiply X and W as in w1*x1 + w2*x2 and we add bias (i.e.constant C to this )
    model_equation = tf.matmul(X1, w1) + tf.matmul(X2, w2) + b

    # remember cost function equation ? ->  (1/2m) summation(from 1 to m) of (hx - y)^2 and objective is to
    # minimize it
    # reduce_sum - calculates the sum of elements across dimensions. Since we will be pumping data for x1, x2 and
    # real value y,it will keep calculating the sum for these equation for each of the values and sum it up.
    # The 'summation' from cost function, this is what reduce_sum does.
    cost_function = tf.reduce_sum(tf.pow(model_equation - Y, 2))/(2 * len(gdp_data)) # len(gdp_data) tells the size of input data i.e. m

    # we will take step of 0.01 size and want this cost function to be minimized i.e.
    # we want the difference between sample (training) set and real value to be minimal
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost_function)

    initState = tf.global_variables_initializer() # initialize variables, we had set weight and bias in the graph above

    # magic starts
    with tf.Session() as session:
        cost = 0.0
        session.run(initState)
        for iteration in range(20):
            for(x1, x2, y) in zip(gdp_data, interest_rate_data, sales_data):
                session.run(optimizer, feed_dict = {X1: x1, X2: x2, Y: y}) # here we feed data
                newcost = session.run(cost_function, feed_dict={X1: gdp_data, X2: interest_rate_data, Y: sales_data})
                print "cost : " + str(newcost) + ",W1=" + str(session.run(w1)) + \
                      ",W2=" + str(session.run(w2)) + ",c=" + str(session.run(bias))

        return str(session.run(w1)), str(session.run(w2)),  str(session.run(bias))

def main():
    read_data()
    w1, w2, c = createModel()
    print "values of w1 : " + str(w1) + ", w2: " + str(w2) + ", c: " + str(c)
    draw_graph()

main()

# refernce - https://blog.altoros.com/using-linear-regression-in-tensorflow.html
