import random
from math import exp


def assign_random_weights(inputs, neuron_hidden, outputs):
    weight_list = []
    input_hidden = []
    b = []

    for i in range(neuron_hidden):
        b = {'weights': [random.uniform(0, 1)
                         for input in range(inputs+1)]}
        input_hidden.append(b)
        b = []

    weight_list.append(input_hidden)
    input_hidden = []

    for i in range(outputs):

        b = {'weights': [random.uniform(0, 1)
                         for input in range(neuron_hidden+1)]}
        input_hidden.append(b)
        b = []

    weight_list.append(input_hidden)

    return weight_list


def summation(weights, inputs):
    summed_input = weights[-1]
    for i in range(len(weights)-1):
        summed_input += weights[i]*float(inputs[i])
    return summed_input


def sigmoid_derivative(input):
    return input*(1.0-input)


def sigmoid(summed_input):
    return 1.0 / (1.0 + exp(-summed_input))


def forward_propagation(weight_list, input):

    inputs = input
    for single_list in weight_list:
        updated_inputs = []
        for perceptron in single_list:
            summed_input = summation(perceptron['weights'], inputs)
            perceptron['output'] = sigmoid(summed_input)
            updated_inputs.append(perceptron['output'])
        inputs = updated_inputs
    return inputs


def back_propagation(weight_list, expected_output):

    for i in reversed(range(len(weight_list))):
        layer = weight_list[i]
        errors = []
        if i != len(weight_list)-1:
            for j in range(len(layer)):
                error = 0.0
                for perceptron in weight_list[i + 1]:
                    error += (perceptron['weights'][j] * perceptron['delta'])
                errors.append(error)

        else:
            for j in range(len(layer)):
                perceptron = layer[j]
                errors.append(expected_output[j] / perceptron['output'])

        for j in range(len(layer)):
            perceptron = layer[j]
            perceptron['delta'] = errors[j] * \
                sigmoid_derivative(perceptron['output'])


def update_weight(weight_list, input):
    eta = 0.001
    for i in range(len(weight_list)):
        inputs = input[:-1]
        if i != 0:
            inputs = [perceptron['output'] for perceptron in weight_list[i-1]]

        for perceptron in weight_list[i]:
            for j in range(len(inputs)):
                perceptron['weights'][j] += eta * \
                    perceptron['delta']*float(inputs[j])
            perceptron['weights'][-1] += eta * perceptron['delta']


def accuracy(data, weight_list):
    accuracy_1 = 0
    i = 0
    for row in data:
        i = i+1
        output = forward_propagation(weight_list, row)
        output_class = output.index(max(output))+1
        if output_class == int(row[-1]):
            accuracy_1 = accuracy_1 + 1
    return (accuracy_1/float(i))*100


arr2 = []
file = open('/home/deepesh/Downloads/Colon_Cancer_CNN_Features.csv', 'r')
for line in file:
    arr1 = line.rstrip("\r\n").split(',')
    arr2.append(arr1)
file.close()
random.shuffle(arr2)
training_dataset = arr2[:5218]
testing_dataset = arr2[5218:]

for j in range(5, 16):

    weight_list = assign_random_weights(325, j, 4)

    for i in range(3):
        for row in training_dataset:

            forward_propagation(weight_list, row)
            expected_output = [0 for i in range(4)]
            expected_output[int(row[-1])-1] = 1
            back_propagation(weight_list, expected_output)
            update_weight(weight_list, row)

    a = accuracy(testing_dataset, weight_list)
    print(j, a)
