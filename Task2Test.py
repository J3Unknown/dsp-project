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



def SinCosSignalSamplesAreEqual(user_choice, file_name, indices, samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
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

    if len(expected_samples) != len(samples):
        print(user_choice + " Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print(user_choice + " Test case failed, your signal have different values from the expected one")
            return
    print(user_choice + " Test case passed successfully")


def SubSignalSamplesAreEqual(userFirstSignal, userSecondSignal, Your_indices, Your_samples):
    if userFirstSignal == 'Signal1.txt' and userSecondSignal == 'Signal2.txt':
        file_name = "tests/task2/signal1-signal2.txt"  # write here path of signal1-signal2
    elif userFirstSignal == 'Signal1.txt' and userSecondSignal == 'Signal3.txt':
        file_name = "tests/task2/signal1-signal3.txt"  # write here path of signal1-signal3

    expected_indices, expected_samples = ReadSignalFile(file_name)

    if (len(expected_samples) != len(Your_samples)) or (len(expected_indices) != len(Your_indices)):
        print("Subtraction Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            print("Subtraction Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Subtraction Test case failed, your signal have different values from the expected one")
            return
    print("Subtraction Test case passed successfully")


def NormalizeSignal(MinRange, MaxRange, Your_indices, Your_samples):
    if MinRange == -1 and MaxRange == 1:
        file_name = "/tests/task2/normalize of signal 1 (from -1 to 1)-- output.txt"
    elif MinRange == 0 and MaxRange == 1:
        file_name = "/tests/task2/normlize signal 2 (from 0 to 1 )-- output.txt"

    expected_indices, expected_samples = ReadSignalFile(file_name)

    if (len(expected_samples) != len(Your_samples)) or (len(expected_indices) != len(Your_indices)):
        print("Normalization Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            print("Normalization Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Normalization Test case failed, your signal have different values from the expected one")
            return
    print("Normalization Test case passed successfully")


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
                line = f.readline()
                continue
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
