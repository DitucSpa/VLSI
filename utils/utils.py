import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import time
import math
from prettytable import PrettyTable
from statistics import stdev
from itertools import combinations
from z3 import *

def GetValues(line):
    return re.findall(r'\d+', line)

# load the instance from the file
def LoadInstance(path, filename):
    values = []
    with open(path+filename) as f:
        for line in f:
            values.append(GetValues(line))
        f.close()
    return BuildInstance(values)

# from file to dict
def BuildInstance(values):
    instance = {'w':int(values[0][0]), 'n_rectangles':int(values[1][0])}
    x_components = []
    y_components = []
    for i in range(instance['n_rectangles']):
        x_components.append(int(values[2+i][0]))
        y_components.append(int(values[2+i][1]))
    instance['x_components']=x_components
    instance['y_components']=y_components
    area = []
    for i in range(len(x_components)):
        area.append(x_components[i]*y_components[i])
    instance['min_h'] = math.ceil(sum(area)/instance['w'])
    return instance

# plot the instance
def PlotInstance(instance, path=None, name='figure'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap('Pastel1')

    # define the colors
    number=10
    colors = [cmap(i) for i in np.linspace(0, 1, number)]

    ax.grid(linewidth=0.5, linestyle='-')
    fontsize = "15" if len(instance['x_positions'])<20 else '5'

    for i in range(len(instance['x_positions'])):
        rect = matplotlib.patches.Rectangle((instance['x_positions'][i], instance['y_positions'][i]),
                                             instance['x_components'][i], instance['y_components'][i],
                                             edgecolor ='black', facecolor=colors[(i+1)%7])
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width()/2.0
        cy = ry + rect.get_height()/2.0
        ax.annotate(i+1, (cx, cy), color='black', weight='bold', fontsize=fontsize, ha='center', va='center')
    h = instance['h']
    w = instance['W']
    plt.ylim([0, h+h*0.2])
    plt.xlim([0, w+w*0.2])
    if h>20 and h<50:
        plt.yticks(np.arange(0, h+h*0.2, 2.0))
        plt.xticks(np.arange(0, w+w*0.2, 2.0))
    elif h>50 and h<100:
        plt.yticks(np.arange(0, h+h*0.2, 5.0))
        plt.xticks(np.arange(0, w+w*0.2, 5.0))
    else:
        plt.yticks(np.arange(0, h+h*0.2, 1.0))
        plt.xticks(np.arange(0, w+w*0.2, 1.0))
    if path:
        plt.savefig(path+'/'+name+'.png')
    else:
        plt.show(block=False)
    plt.close()

# write the instance
def WriteInstance(instance, path=None, name='out-1'):
    if path: path += '/'
    with open(path + name + '.txt', 'w') as f:
        f.write('{0} {1}'.format(instance['W'], instance['h']))
        n = len(instance['x_components'])
        f.write('\n{0}'.format(n))
        for i in range(n):
            f.write('\n{0} {1} {2} {3}'.format(instance['x_components'][i],
                                                   instance['y_components'][i],
                                                   instance['x_positions'][i],
                                                   instance['y_positions'][i]))


def Statistics(models, optimal_h, timeout=300):
    columns = []
    for key in list(models.keys()):
        models[key] = models[key].apply(lambda x: timeout if x=='TIMEOUT' else float(x))
        optimal_h[key] = optimal_h[key].apply(lambda x: -1 if x=='DNF' else float(x))
    t = PrettyTable(['',] + list(models.keys()))

    # compute total time
    total_time = []
    for key in list(models.keys()):
        total_time.append(round(models[key].sum(),3))
    t.add_row(['Total Time [s]',]+total_time)

    # compute max
    maxim = []
    for key in list(models.keys()):
        maxim.append(round(max(models[key]),3))
    t.add_row(['Max [s]',]+maxim)

    # compute min
    minim = []
    for key in list(models.keys()):
        minim.append(round(min(models[key]),3))
    t.add_row(['Min [s]',]+minim)

    # compute mean
    mean = []
    for key in list(models.keys()):
        mean.append(round(models[key].sum()/len(list(models[key])),3))
    t.add_row(['Mean [s]',]+mean)

    # compute std
    std = []
    for key in list(models.keys()):
        std.append(round(stdev(list(models[key])),3))
    t.add_row(['Std [s]',]+std)

    # compute instances solved
    solved = []
    for key in list(models.keys()):
        solved.append(str(models[key][models[key]!=timeout].count())+'/'+str(len(list(models[key]))))
    t.add_row(['Instances Solved',]+solved)

    # compute the optimal h
    optimal = []
    h_min = optimal_h['h_min']
    for key in list(models.keys()):
        optimal.append(str(optimal_h[key][optimal_h[key]==h_min].count())+'/'+str(len(list(models[key]))))
    t.add_row(['Optimal H',]+optimal)

    for i in list(models.keys()):
        t.align[i]='r'

    print('TIMEOUT is seen as 300s.\n')
    print(t)

def PlotStats(models, x_axis, timeout=300, width_bar=0.3, figsize=(15,10)):
    fig = plt.figure(figsize=figsize)
    left, bottom, width, height = 0.1, 0.3, 0.8, 0.6
    ax = fig.add_axes([left, bottom, width, height])
    width = width_bar
    ticks = np.arange(1,len(x_axis)+1)
    wid = 0
    for key in list(models.keys()):
        models[key] = models[key].apply(lambda x: timeout if x=='TIMEOUT' else float(x))
        models[key]=[k if k!=timeout else 0 for k in list(models[key])]
        ax.bar(ticks+wid, models[key], width, label=key)
        wid += width
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Instances')
    ax.set_yticks(ticks + width/2)*2
    ax.set_title(' VS '.join([key for key in list(models.keys())]), fontsize=12)
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.xaxis.grid(True, linestyle='-', color='black', alpha = 0.2)
    ax.yaxis.grid(True, linestyle='-', color='black', alpha = 0.2)
    plt.show()

def PlotResults(models, x_axis, timeout=300, width_bar=0.3, figsize=(15,10)):
    fig = plt.figure(figsize=figsize)
    left, bottom, width, height = 0.1, 0.3, 0.8, 0.6
    ax = fig.add_axes([left, bottom, width, height])
    width = width_bar
    ticks = np.arange(1,len(x_axis)+1)
    wid = 0
    for key in list(models.keys()):
        models[key] = models[key].apply(lambda x: timeout if x=='TIMEOUT' else float(x))
        models[key]=[k if k!=timeout else 0 for k in list(models[key])]
        ax.barh(ticks+wid, models[key], width, label=key)
        wid += width
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Instances')
    ax.set_xticks(ticks + width/2)*2
    ax.set_title(' VS '.join([key for key in list(models.keys())]), fontsize=12)
    ax.set_xscale('log')
    ax.legend(loc='best')
    ax.xaxis.grid(True, linestyle='-', color='black', alpha = 0.2)
    ax.yaxis.grid(True, linestyle='-', color='black', alpha = 0.2)
    plt.show()

def at_least_one(bool_vars):
    return Or(bool_vars)

def at_most_one(bool_vars):
    return [Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)]

def exactly_one(bool_vars):
    return at_most_one(bool_vars) + [at_least_one(bool_vars)]

# the function counts the number of elements for each model
def CountElement(series, models):
    unique = set(series)
    counts = {}
    for i in list(unique):
        counts[i] = series.eq(i).sum()
    for i in models:
        if i not in list(counts.keys()):
            counts[i] = 0
    return counts

# this function is used to get the max-min in each row for the results dataset containing the 4 models
def Results(dataset, models):

    instances = dataset.shape[0]

    # first we compute the max-min of the first half instances
    maximum_first_half = CountElement(dataset.iloc[:int(instances/2)].idxmax(axis=1), models)
    minimum_first_half = CountElement(dataset.iloc[:int(instances/2)].idxmin(axis=1), models)
    timeout_first_half = dict(dataset.iloc[:int(instances/2)].eq(0).sum())

    # then, we compute the second half
    maximum_second_half = CountElement(dataset.iloc[int(instances/2):].idxmax(axis=1), models)
    minimum_second_half = CountElement(dataset.iloc[int(instances/2):].idxmin(axis=1), models)
    timeout_second_half = dict(dataset.iloc[int(instances/2):].eq(0).sum())

    # create the table
    t = PrettyTable(['',] + models)
    t.add_row(['Maximums (First Half)',]+[str(maximum_first_half[k])+'/'+str(int(instances/2)) for k in models])
    t.add_row(['Minimums (First Half)',]+[str(minimum_first_half[k])+'/'+str(int(instances/2)) for k in models])
    t.add_row(['Timeout (First Half)',]+[str(timeout_first_half[k])+'/'+str(int(instances/2)) for k in models])
    t.add_row([""]*(len(models)+1))

    t.add_row(['Maximums (Second Half)',]+[str(maximum_second_half[k])+'/'+str(int(instances/2)) for k in models])
    t.add_row(['Minimums (Second Half)',]+[str(minimum_second_half[k])+'/'+str(int(instances/2)) for k in models])
    t.add_row(['Timeout (Second Half)',]+[str(timeout_second_half[k])+'/'+str(int(instances/2)) for k in models])
    t.add_row([""]*(len(models)+1))

    t.add_row(['Total Minimums',]+[str(minimum_second_half[k]+minimum_first_half[k])+'/'+str(instances) for k in models])
    t.add_row(['Total Maximums',]+[str(maximum_first_half[k]+maximum_second_half[k])+'/'+str(instances) for k in models])
    t.add_row(['Total Timeout',]+[str(timeout_first_half[k]+timeout_second_half[k])+'/'+str(instances) for k in models])

    for i in list(maximum_first_half.keys()):
        t.align[i]='r'

    print(t)


def highlight_max(row):
    is_max = row == row.max()
    if len([x for x in row.values if x > 0])==1:
        return ['background-color: #90EE90' if v else '' for v in is_max]
    return ['background-color: #F4F410' if v else '' for v in is_max]

def highlight_min(row):
    is_min = row == row[row!=0].min()
    return ['background-color: #90EE90' if v else '' for v in is_min]

def highlight_timeout(row):
    is_min = row == row[row==0].min()
    return ['background-color: #F08080' if v else '' for v in is_min]
