import json
import os
import re
from tokenize import Number

import pandas as pd
#(os.path.dirname(os.path.dirname(os.getcwd()))

current =os.path.dirname(os.path.dirname(os.getcwd()))
print(current)
folders = ["dogs","Cifar","poke","brain"]
for folder in folders:
    path =os.path.join(current,f"Python/RP/logs/{folder}/Txt/")
    fileNames = os.listdir(path)
    print(path)

    files = dict()
    fileEpochs = dict()
    for name in fileNames:
        try:
            with open(os.path.join(path,name),"r") as f:
                files[name] = f.read()
                f.close()
        except:
            with open(os.path.join(path,name),"r",encoding = "utf8") as f:
                files[name] = f.read()
                f.close()
    print(files.keys())
    for name in files.keys():
        file = files[name]
        if "Model3" in file:
            fileEpochs[name] = {}
            file = file[file.index("==================== Epoch:"):]
            epochs = file.split("==================== Epoch:")

            for  e in epochs[1:]:
                epoch  = e.split('\n')
                numb = list(re.findall("(\d+)", epoch[0]))
                epoch_Number = int(numb[0])
                fileEpochs[name][epoch_Number] = {"Epoch Number ": epoch_Number}
                for x in epoch[1:]:
                    for y in x.split("|"):
                        if "Train acc" in y:
                            numb = list(re.findall("(\d+)", y))
                            fileEpochs[name][epoch_Number]["Train Accuracy"] = (int(numb[0]))+(float(numb[1]) * 10 ** -len(numb[1]))
                        elif "Validation acc" in y:
                            numb = list(re.findall("(\d+)", y))
                            fileEpochs[name][epoch_Number]["Test Accuracy"] = (int(numb[0]))+(float(numb[1]) * 10 ** -len(numb[1]))
                        elif "Validation Loss" in y:
                            numb = list(re.findall("(\d+)", y))
                            fileEpochs[name][epoch_Number]["Loss"] = (int(numb[0]))+(float(numb[1]) * 10 ** -len(numb[1]))
                        elif "Train Loss" in y:
                            pass
                        elif "Time" in y:
                            numb = list(re.findall("(\d+)", y))
                            fileEpochs[name][epoch_Number]["Time"] = (int(numb[0]))+(float(numb[1]) * 10 ** -len(numb[1]))
        elif "Model2" in file:
            fileEpochs[name] = {}
            file = file[file.index("Epoch "):]

            indexT2 = file.index("Validation loss: ")
            epochs = file[:indexT2].split("Epoch ")
            indexT1 = file.rindex("Epoch ")
            for  e in epochs[1:]:
                epoch  = e.split('\n')
                numb = list(re.findall("(\d+)", epoch[0]))
                epoch_Number = int(numb[0])
                fileEpochs[name][epoch_Number] = {"Epoch Number ": epoch_Number}
                for x in epoch[1:]:
                    if "[" in x:
                        timeL = json.loads(x)
                        print(timeL,timeL.__class__)
                        for t in timeL:
                            fileEpochs[name][timeL.index(t)+1]["Time"] = t
                    elif "dense" in x:
                        for y in x.split("-"):
                            if "val_dense_3_accuracy" in y:
                                numb = list(re.findall("(\d+)", y))
                                fileEpochs[name][epoch_Number]["Test Accuracy dense3"] = ((int(numb[1]))+(float(numb[2]) * 10 ** -len(numb[2])))
                            elif "val_dense_1_accuracy" in y:
                                numb = list(re.findall("(\d+)", y))
                                fileEpochs[name][epoch_Number]["Test Accuracy dense1"] = ((int(numb[1]))+(float(numb[2]) * 10 ** -len(numb[2])))
                            elif "val_dense_3_loss" in y:
                                numb = list(re.findall("(\d+)", y))
                                fileEpochs[name][epoch_Number]["Test loss d3"] = ((int(numb[1]))+(float(numb[2]) * 10 ** -len(numb[2])))
                            elif "val_dense_1_loss" in y:
                                numb = list(re.findall("(\d+)", y))
                                fileEpochs[name][epoch_Number]["Test loss d1"] = ((int(numb[1]))+(float(numb[2]) * 10 ** -len(numb[2])))
                            elif "val_loss" in y:
                                numb = list(re.findall("(\d+)", y))
                                fileEpochs[name][epoch_Number]["Test loss"] = ((int(numb[0])) + (float(numb[1]) * 10 **  -len(numb[1])))
        else:
            if "Stage 1" in file:
                file = file[file.index("Stage1, Epoch "):]
                epochs = file.split("Stage1, Epoch ")
            elif "Stage 2" in file:
                file = file[file.index("Stage2, Epoch "):]
                epochs = file.split("Stage2, Epoch ")
            else:
                file = file[file.index("Epoch "):]
                epochs = file.split("Epoch ")
            fileEpochs[name]={}
            for e in epochs[1:]:
                if "Test Accuracy: tensor" in e:
                    splitAt = e.index("Test Accuracy: tensor")
                    epoch = e[:splitAt].split("\n")
                else:
                    epoch = e.split("\n")
                numb = list(re.findall("(\d+)", epoch[0]))
                epoch_Number = int(numb[0])
                fileEpochs[name][epoch_Number] = {"Epoch Number ":epoch_Number}
                fileEpochs[name][epoch_Number]["Time"] = int(numb[1]) * 60 + int(numb[2])
                #fileEpochs[name][epoch_Number]["Time"] = pd.Timedelta(int(numb[1]) * 60 + int(numb[2]), unit="s")

                if "Test Accuracy: tensor" in e:
                    e2 = e[splitAt:].replace("\t", "").replace("\n", "")
                    s = json.loads(e2[e2.index("["): e2.index("]") + 1])
                    device = e2[e2.index("device"):e2.index("')") + 1].split("=")
                    avg = sum(s) / len(s)

                    fileEpochs[name][epoch_Number]["Test Accuracy Tensor:"] = s
                    fileEpochs[name][epoch_Number]["Test Accuracy avg:"] = avg
                    fileEpochs[name][epoch_Number]["device"] = device[1].replace("\'","")

                for line in epoch[1:]:
                    for part in line.split(","):
                        if "Train Accuracy:" in part:
                            numb = list(re.findall("(\d+)", part))
                            fileEpochs[name][epoch_Number]["Train Accuracy"] = float(numb[1])*10**-len(numb[1])
                        elif "Test Accuracy:" in part:
                            numb = list(re.findall("(\d+)", part))
                            fileEpochs[name][epoch_Number]["Test Accuracy"] = float(numb[1])*10**-len(numb[1])
                        elif "Train Loss" in part:
                            pass
                        elif "Test spiking time:" in part:
                            pass
                            #numb = list(re.findall("(\d+)", part))
                            #fileEpochs[name][epoch_Number]["Test spiking time"] = ((int(numb[0]))+(float(numb[1]) * 10 ** -len(numb[1])))
                        elif "Loss:" in part:
                            numb = list(re.findall("(\d+)", part))
                            fileEpochs[name][epoch_Number]["Loss"] = float(numb[1]) * 10 ** -len(numb[1])
    for name in fileEpochs.keys():
        df = pd.DataFrame(fileEpochs[name]).T
        #print(df)
        #df['Time'] = df['Time']
        df.to_csv(os.path.join(path.replace("Txt","csv"),name.replace(".txt",".csv")))
        df.to_excel(os.path.join(path.replace("Txt","xlsx"),name.replace(".txt", ".xlsx")))



