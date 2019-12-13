from csv import reader
import sys
import random
import MathAndStats as ms
import FeedforwardNetwork as ffn
import EvolutionaryAlgorithms as ea

def openFile(data_file):
    lines = open(data_file, "r").readlines()
    csv_lines = reader(lines)
    data = []

    for line in csv_lines:
        tmp = []
        # remove line number from each example (first column)
        for c in range(1, len(line) - 1):
            tmp.append(float(line[c]))
        if sys.argv[2] == 'r':
            tmp.append(float(line[-1]))
        else:
            tmp.append(line[-1])
        data.append(tmp)

    data = ms.normalize(data)
    # divide data into 10 chunks for use in 10-fold cross validation paired t test
    chnks = getNChunks(data, 10)
    class_list = getClasses(data)
    return chnks, class_list

# divide the example set into n random chunks of approximately equal size
def getNChunks(data, n):
    # randomly shuffle the order of examples in the data set
    random.shuffle(data)
    dataLen = len(data)
    chunkLen = int(dataLen / n)
    # chunks is a list of groups of examples
    chunks = []
    # rows are observation
    # columns are labels

    # group the examples in data into chunks
    for obs in range(0, dataLen, chunkLen):
        if (obs + chunkLen) <= dataLen:
            chunk = data[obs:obs + chunkLen]
            chunks.append(chunk)
    # append the extra examples randomly to the chunks
    for i in range(n*chunkLen, dataLen):
        chunks[random.randint(0,n-1)].append(data[i])
    for i in range(len(chunks)):
        print("Length of chunk: ", len(chunks[i]))
    return chunks

def getClasses(data):
    if sys.argv[2] == 'r':
        return []
    classes = []
    for obs in range(len(data)):
        if not data[obs][-1] in classes:
            classes.append(data[obs][-1])
    return classes

def tenFoldCV(chunked_data, clss_list, use_regression, hidden_layer_nodes):
    for test_num in range(10):
        print("\n\n\nFold: ", test_num+1, "of 10 fold cross validation")
        training_set = []

        testing_set = chunked_data[test_num]
        # make example set
        for train in range(10):
            if train != test_num:
                for x in range(len(chunked_data[train])):
                    training_set.append(chunked_data[train][x])

        validation_index = int((float(len(training_set)) * 8 / 10)) - 1
        if use_regression:
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            mlp.setWeights([len(chunks[0][0]) - 1] + hidden_layer_nodes[:1],
                           ea.generateFFNGA(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:1],
                             prob_cross=0.6, prob_mutation=0.01, mutation_variance=0.2, population_size=40,
                             max_generations=1000))
            results = ms.testRegressor(mlp, testing_set)
            print(ms.getMean(results, len(results)))


if(len(sys.argv) > 2):
    chunks, class_list = openFile(sys.argv[1])
    uses_regression = False
    if sys.argv[2] == 'r':
        print("Using regression")
        uses_regression = True
    else:
        print("Using classification")

    hidden_layer_nodes = []
    for i in range(3):
        hidden_layer_nodes.append(4 * (len(chunks[0][0]) - 1))
    tenFoldCV(chunks, class_list, uses_regression, hidden_layer_nodes)

else:
    print("Usage: <datafile> <r/c>")