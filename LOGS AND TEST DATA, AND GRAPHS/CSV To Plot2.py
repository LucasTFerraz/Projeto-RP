import itertools
import json
import os
import random
import re
from random import randint
from tokenize import Number
from datetime import timedelta
import pandas as pd
from matplotlib import ticker
import seaborn as sns

def pad_dict_list(dict_list):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))

    for lname in dict_list.keys():
        padel = max(dict_list[lname])
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list

def format_func(x, pos):
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    seconds = int(x % 60)

    return "{:d}:{:02d}:{:02d}".format(hours, minutes,seconds)

#(os.path.dirname(os.path.dirname(os.getcwd()))
current =os.path.dirname(os.path.dirname(os.getcwd()))
#print(current)
folders = ["dogs","cifar","Poke","brain"]
cat_values = {"accuracy": {g:{} for g in folders},
        "TotalTime" : {g:{} for g in folders},
        "Loss" : {g:{} for g in folders},
        "epoch_time_max":{g:{} for g in folders},
        "epoch_time_mean": {g:{} for g in folders},
        "epoch_time_median": {g:{} for g in folders},
        "epoch_time_min": {g:{} for g in folders},
        "epoch_time_SD": {g:{} for g in folders},
        "EpochNumber" : {g:{} for g in folders}}
for group in folders:

    path = os.path.join(current,f"Python/RP2/Logs/{group}/CSV/")
    fileNames = os.listdir(path)
    values={

        "accuracy": {},
        "TotalTime" : {},
        "Loss" : {},
        "epoch_time_max": {},
        "epoch_time_mean": {},
        "epoch_time_median": {},
        "epoch_time_min": {},
        "epoch_time_SD": {},
        "EpochNumber" : {}}
    values_C = {
        "accuracies": {},
        "Time": {},
        "Loss": {}}

    files = dict()
    fileEpochs = dict()
    for name in fileNames:
        files[name] = pd.read_csv(os.path.join(path,name))
    for name,df in files.items():

        if "Model2" in name:
            name = "Model Brain"
            a = df["Test Accuracy dense3"].max()
            values["accuracy"][name] = a
            l = df["Test Accuracy dense3"].tolist()
            values_C["accuracies"][name] = l
            values["Loss"][name] = df["Test loss d3"][l.index(a)]
            values_C["Loss"][name] = df["Test loss d3"].tolist()
        else:
            if "Model_3" in name:
                name = "Model Dog"
            try:
                a = df["Test Accuracy"].max()
                l = df["Test Accuracy"].tolist()
            except:
                l1 = [json.loads(s) for s in df["Test Accuracy Tensor:"].tolist()]
                l = [x[-1] for x in l1]
                a = max(l)
            values["accuracy"][name] = a
            values_C["accuracies"][name] = l
            values["Loss"][name] = df["Loss"][l.index(a)]
            values_C["Loss"][name] = df["Loss"].tolist()
        values_C["Time"][name] = df["Time"].tolist()
        values["TotalTime"][name] = df["Time"].sum(numeric_only=False)
        values["epoch_time_mean"][name] = df["Time"].mean(numeric_only=False)
        values["epoch_time_median"][name] = df["Time"].median(numeric_only=False)
        values["epoch_time_max"][name] = df["Time"].max(numeric_only=False)
        values["epoch_time_min"][name] = df["Time"].min(numeric_only=False)
        values["epoch_time_SD"][name] = df["Time"].std(numeric_only=False)
        values["EpochNumber"][name] = len(df.index)
    import matplotlib.pyplot as plt

    """for v,d in values.items():
        l = []
        h = []
        #https://stackoverflow.com/questions/56775785/assign-color-using-seaborn-based-on-column-name
        for i,j in d.items():
            i = i.replace(".csv","").replace("log","")
            i = i.replace("ResNet","RsN").replace("log","")
            i = i.replace("Main", "M").replace("Base", "B")
            l.append(i.replace(".csv","").replace("log",""))
            h.append(j)


        palette = {p:(0.99, 0.0, 0.0) for p in l}
        for p in l:
            if "B" in p:
                palette[p] = (0.0, 0.0, 0.0)
            elif "1" in p:
                if "16" in p and "2" not in p:
                    palette[p] = (0.99, 0.50, 0.0)
                elif "20" in p:
                    palette[p] = (0.99, 0.0, 0.50)
        palette["Model Brain"] = (0.0,0.99, 0.0)
        palette["Model Dog"] = (0.0, 0.0, 0.99)
        data = pd.DataFrame({"Model": l, v: h})

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=data,x="Model", y=v,palette=palette)

        for container in ax.containers:
            ax.bar_label(container)
        if "accuracy" in v.lower():
            ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        elif "time" in v.lower():
            pass
        '''for i in data:
            sns.barplot(x=data.index, y=i, data=data, color=palette[i])'''
        plt.title(f"Grafico : {v}")
        #plt.savefig(f'Logs Img/{group}-{v}bar.png')
        #plt.show()
        data.to_csv(f"Logs Img/Table-{group}.csv")"""
    """for v, d in values_C.items():
        for v, d in values.items():
            l = []
            h = []
            '''line = "accuracies" in v
            sns.regplot(x="x", y="f", data=df1, order=2, ax=ax)
            sns.regplot(x="x", y="g", data=df2, order=2, ax=ax2)
            for container in ax.containers:
                ax.bar_label(container)
            if "accuracy" in v.lower():
                ax.yaxis.set_major_formatter(ticker.PercentFormatter())
            elif "time" in v.lower():
                pass
            plt.title(f"Grafico : {v}")
            plt.savefig(f'Logs Img/{group}-{v}bar.png')'''
            for x,y in values_C.items():
                y = pad_dict_list(y)
                data = pd.DataFrame(y)
                data.to_excel(f"Table -{group} -{x}.xlsx")
            # plt.show()w'''"""
    values2 = {x:{} for x in values['accuracy'].keys()}
    for x in values.keys():
        print(x)
        cat_values[x][group] = values[x]
        for y in values2.keys():

            values2[y][x] = values[x][y]
    #print(values)
    #print(values2)
    #data = pd.DataFrame(values2).T
    #plt.clf()
    #fig, ax2 = plt.subplots()
    # hide axes
    #data["TotalTime"] =  pd.to_timedelta(data["TotalTime"], unit='s').astype(str).str.split(' ').str[-1]
    #data["epoch_time_mean"] = pd.to_timedelta(data["epoch_time_mean"], unit='s').astype(str).str.split(' ').str[-1]

    #fig.patch.set_visible(False)
    #print(data.index)
    #print(list(data.index))
    #ax2.axis('off')
    #ax2.axis('tight')
    #ax2.table(cellText=data.values,colLabels=data.columns, loc='center',rowLabels=list(data.index))
    #fig.tight_layout()
    #plt.savefig(f'Logs Img/{group}-Table.png',dpi=500)
    #data.to_excel(f"Table-{group}.xlsx")
for x in cat_values.keys():
    if "EpochNumber" == x:
        continue
    if "accuracy" not in x:
        continue
    d = {"Model":[],"Dataset":[],x:[]}
    acu =  "accuracy" in x.lower()
    y = cat_values[x]
    l = y.keys()
    print(cat_values)
    for j in l:
        for i in y[j].keys():
            if "Main1" not in i:
                i2 = i.replace(".csv", "").replace("log", "")
                i2 = i2.replace("ResNet", "RsN").replace("log", "")
                i2 = i2.replace("Main", "M").replace("Base", "B")
                d["Model"].append(i.replace(".csv", "").replace("log", ""))
                d["Dataset"].append(j)
                if acu:
                    d[x].append(100*round(y[j][i],3))
                else:
                    d[x].append(y[j][i])
    palette_b = [(0, 76, 153),(204, 0, 0),(0, 153, 76),(255, 128, 0),
                 (102, 0, 204),(0, 204, 204),(255, 204, 0),(128, 128, 128),
                 (255,51,153)]
    for i in range(len(palette_b)):
        a,b,c = palette_b[i]
        palette_b[i]=(a/255,b/255,c/255)

    palette = {p: random.choice(palette_b[2]) for p in d["Model"]}
    palette["Model Brain"] = palette_b[1]
    palette["Model Dog"] =  palette_b[0]
    data = pd.DataFrame(d)

    #print(data)

    g = sns.catplot(x='Dataset', y=x, hue='Model', data=data, kind='bar',
                    height=5, aspect=1, palette=palette,width= 0.95 )
    g.figure.set_size_inches(12,8)
    ax = g.facet_axis(0, 0)
    bottom, top = ax.get_ylim()
    print(bottom)
    for container in ax.containers:
        ax.bar_label(container)
    if acu:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    elif "time" in x.lower():
        pass
    data.to_excel(f"Table -ALL -{x}.xlsx")
    plt.title(f"Grafico : {x}")
    plt.savefig(f'Logs Img/ALL-{x}bar.png')#,dpi=300)
    data.to_excel(f"Logs Img/Table-{x}.xlsx")



