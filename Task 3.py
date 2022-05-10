import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Data = pd.read_csv("Task3 - Dataset - HIV RVG.csv")
print(Data)

Data = Data.drop("Image number",1)
DataSum = Data.drop("Participant Condition",1)

#get mean
print("Mean of data")
mean = DataSum.mean(axis=0)
print(mean)

#get max
print("Max of data")
max = DataSum.max(axis=0)
print(max)

#get min
print("Min of data")
min = DataSum.min(axis=0)
print(min)

#get standard div
print("Standard Deviation of data")
std = DataSum.std(axis=0)
print(std)

#splits the data by control and patient only carrying the alpha column
Control = (Data.loc[Data['Participant Condition'] == 'Control', 'Alpha'])
Patient = (Data.loc[Data['Participant Condition'] == 'Patient', 'Alpha'])

#plots a graph for the aplha data
fig, ax =plt.subplots()
ax.boxplot([Control,Patient], labels=['Control','Patient'])
plt.title("BoxPlot for Alpha")
plt.ylabel("Aplha")
plt.xlabel("Participant Condition")
plt.show()

ControlDens = (Data.loc[Data['Participant Condition'] == 'Control', 'Beta'])
PatientDens = (Data.loc[Data['Participant Condition'] == 'Patient', 'Beta'])

sns.kdeplot(ControlDens,label='Control',fill=True)
sns.kdeplot(PatientDens,label='Patient',fill=True,bw=0.5)
plt.legend(labels=['Control','Patient'])
plt.show()