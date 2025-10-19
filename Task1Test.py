#!/usr/bin/env python
# coding: utf-8

def ReadSignalFile(file_name):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        for line in f:
            L = line.strip()
            if len(L.split()) == 2:
                V1, V2 = L.split()
                expected_indices.append(int(V1))
                expected_samples.append(float(V2))
            else:
                break
    return expected_indices, expected_samples



def AddSignalSamplesAreEqual(userFirstSignal, userSecondSignal, Your_indices, Your_samples):
    if userFirstSignal == 'Signal1.txt' and userSecondSignal == 'Signal2.txt':
        file_name = "tests/task1/Signal1+signal2.txt"
    elif userFirstSignal == 'Signal1.txt' and userSecondSignal == 'Signal3.txt':
        file_name = "tests/task1/signal1+signal3.txt"

    expected_indices, expected_samples = ReadSignalFile(file_name)
    if (len(expected_samples) != len(Your_samples)) or (len(expected_indices) != len(Your_indices)):
        print("Addition Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            print("Addition Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Addition Test case failed, your signal have different values from the expected one")
            return
    print("Addition Test case passed successfully")


def MultiplySignalByConst(User_Const, Your_indices, Your_samples):
    if User_Const == 5:
        file_name = "tests/task1/MultiplySignalByConstant-Signal1 - by 5.txt"
    elif User_Const == 10:
        file_name = "tests/task1/MultiplySignalByConstant-signal2 - by 10.txt"

    expected_indices, expected_samples = ReadSignalFile(file_name)
    if (len(expected_samples) != len(Your_samples)) or (len(expected_indices) != len(Your_indices)):
        print("Multiply by " + str(User_Const) + " Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            print("Multiply by " + str(User_Const) + " Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Multiply by " + str(User_Const) + " Test case failed, your signal have different values from the expected one")
            return
    print("Multiply by " + str(User_Const) + " Test case passed successfully")


def SignalSamplesAreEqual(TaskName, output_file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(output_file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    if (len(expected_samples) != len(Your_samples)) or (len(expected_indices) != len(Your_indices)):
        print(TaskName + " Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            print(TaskName + " Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print(TaskName + " Test case failed, your signal have different values from the expected one")
            return
    print(TaskName + " Test case passed successfully")
