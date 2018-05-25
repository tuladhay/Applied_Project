import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

#######################################################################################################################
class SaveVal():
    def __init__(self):
        self.testLoss = []
        self.trainLoss = []
        self.trainAcc = []
        self.testAcc = []

# DEFINE THE NEURAL NETWORK
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.num_inputs = 36
        self.num_output = 1
        self.hidden_size = 64
        self.linear1 = nn.Linear(self.num_inputs, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.num_output)
        self.batchNorm = nn.BatchNorm1d(self.hidden_size)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x

def eval_net(inputs, targets):
    correct = 0
    total = 0
    total_loss = 0
    net.eval()
    criterion = nn.MSELoss()

    for i in range(len(inputs)):
        x, y = inputs[i], targets[i]    # They have been already turned into variables
        eval_output = net(x)
        predicted = eval_output.data    # In DL HW3, we needed a max here since it was multiclass classification
        if predicted[0] > 0.5:
            predicted = 1
        else:
            predicted = 0
        total += y.size(0)
        correct += (predicted == y.data[0])
        loss = criterion(eval_output, y)
        total_loss += loss.data[0]
    net.train()  # Why would I do this? -> because the network eval and train mode are different
    return total_loss / total, correct / total


if __name__=="__main__":
    ####################################################################################################################
    #               LOADING AND PRE-PROCESSING THE DATA THAT I HAVE COLLECTED
    ####################################################################################################################
    # TODO: Remove spatial correlation in data
    pickle_off = open("pickled_body_pos_active", "rb")
    active_data = pickle.load(pickle_off)

    pickle_off = open("pickled_body_pos_not_active", "rb")
    not_active_data = pickle.load(pickle_off)

    # REMOVING SPATIAL VARIANCE
    for n in range(len(active_data)):
        # indices 8 and 11 are the right and left hip locations
         hip_avg_x = (active_data[n][0][8] + active_data[n][0][11]) / 2.0  # substracting the average of hip x dist
         hip_avg_y = (active_data[n][1][8] + active_data[n][1][11]) / 2.0  # substracting the average hip height
         active_data[n][0] = [x - hip_avg_x for x in active_data[n][0]]
         active_data[n][1] = [y - hip_avg_y for y in active_data[n][1]]

    for n in range(len(not_active_data)):
        # indices 8 and 11 are the right and left hip locations
         hip_avg_x = (not_active_data[n][0][8] + not_active_data[n][0][11]) / 2.0  # substracting the average of hip x dist
         hip_avg_y = (not_active_data[n][1][8] + not_active_data[n][1][11]) / 2.0  # substracting the average hip height
         not_active_data[n][0] = [x - hip_avg_x for x in not_active_data[n][0]]
         not_active_data[n][1] = [y - hip_avg_y for y in not_active_data[n][1]]

    dataset = active_data + not_active_data
    inputs = []
    targets = []

    for data in dataset:
        inputs.append(data[0] + data[1])
        targets.append(data[2])
        # this is done so that the (x,y) coordinates can be fed together to the neural network

    # Shuffle the dataset
    temp = list(zip(inputs, targets))
    np.random.shuffle(temp)  # this works "in place"
    inputs, targets = zip(*temp)

    targets = torch.FloatTensor(targets)
    inputs = torch.FloatTensor(inputs)

    '''
    Inputs has both x and y coordinates for 18 points
    Target has the label for the data
    Now I have both datasets with labels ready to train
    '''



    ####################################################################################################################
    # Similarly, load the test inputs and targets
    # TODO: Combine into the same file and save it. Then do a 80:20 split while loading
    # TODO: Refactor them into functions?
    active_data = []
    not_active_data = []
    temp = []

    # some wierd encoding this I had to do because this set of data was captured using Python2 pickle,
    # whereas the previous (train) data was captured using Python3 pickle

    # with open("pickled_body_pos_active_test_fixed_error", 'rb') as f:
    #     active_data = pickle.load(f, encoding='latin1')
    with open("pickled_body_pos_not_active_test", 'rb') as f:
         not_active_data = pickle.load(f, encoding='latin1')
        # this part was saved in python2

    pickle_off = open("pickled_body_pos_active_test_fixed_error.p", "rb")
    active_data = pickle.load(pickle_off)
    # This part was originally saved in python2, but I fixed some error and re-saved it using python3


    # REMOVING SPATIAL VARIANCE
    for n in range(len(active_data)):
        # indices 8 and 11 are the right and left hip locations
        hip_avg_x = (active_data[n][0][8] + active_data[n][0][11]) / 2.0  # substracting the average of hip x dist
        hip_avg_y = (active_data[n][1][8] + active_data[n][1][11]) / 2.0  # substracting the average hip height
        active_data[n][0] = [x - hip_avg_x for x in active_data[n][0]]
        active_data[n][1] = [y - hip_avg_y for y in active_data[n][1]]

    for n in range(len(not_active_data)):
        # indices 8 and 11 are the right and left hip locations
        hip_avg_x = (not_active_data[n][0][8] + not_active_data[n][0][
            11]) / 2.0  # substracting the average of hip x dist
        hip_avg_y = (not_active_data[n][1][8] + not_active_data[n][1][11]) / 2.0  # substracting the average hip height
        not_active_data[n][0] = [x - hip_avg_x for x in not_active_data[n][0]]
        not_active_data[n][1] = [y - hip_avg_y for y in not_active_data[n][1]]

    test_dataset = active_data + not_active_data
    test_inputs = []
    test_targets = []

    for data in test_dataset:
        test_inputs.append(data[0] + data[1])
        test_targets.append(data[2])
        # this is done so that the (x,y) coordinates can be fed together to the neural network

    # Shuffle the dataset
    temp = list(zip(test_inputs, test_targets))
    np.random.shuffle(temp)  # this works "in place"
    test_inputs, test_targets = zip(*temp)

    test_targets = torch.FloatTensor(test_targets)
    test_inputs = torch.FloatTensor(test_inputs)
    ####################################################################################################################





    num_epochs = 20
    save = SaveVal()

    net = MLP()
    net.train()
    optimizer = optim.Adam(net.parameters())
    optimizer.zero_grad()
    criterion = nn.MSELoss()

    print("\nStart Training")

    inputs = Variable(inputs)
    targets = Variable(targets)
    test_inputs = Variable(test_inputs)
    test_targets = Variable(test_targets)

    train_loss, train_acc = eval_net(inputs, targets)
    test_loss, test_acc = eval_net(test_inputs, test_targets)
    print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
          (0, train_loss, train_acc, test_loss, test_acc))

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i in range(len(inputs)):
            x, y = inputs[i], targets[i]
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)    # Loss is MSE(output, labels)
            loss.backward()
            optimizer.step()

        # print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(inputs, targets)
        test_loss, test_acc = eval_net(test_inputs, test_targets)

        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch + 1, train_loss, train_acc, test_loss, test_acc))
        save.trainLoss.append(train_loss)
        save.trainAcc.append(train_acc)
        save.testLoss.append(test_loss)
        save.testAcc.append(test_acc)

    pickle.dump(save, open("train_test_results.p", "wb"))
    print("Finished Training")
