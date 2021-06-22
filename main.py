# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Project C117
# %% [markdown]
# ## Getting Data

# %%
import pandas
data_frame = pandas.read_csv('https://raw.githubusercontent.com/whitehatjr/datasets/master/c117/BankNote_Authentication.csv')

x = data_frame["variance"]
y = data_frame["class"]

print(data_frame.head())

# %% [markdown]
# ## Train Test Split

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# %% [markdown]
# ## Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
import numpy

x = numpy.reshape(x_train.ravel(), (len(x_train), 1))
y = numpy.reshape(y_train.ravel(), (len(y_train), 1))

classifier = LogisticRegression(random_state=0)
classifier.fit(x, y)

# %% [markdown]
# ## Prediction

# %%
x_test = numpy.reshape(x_test.ravel(), (len(x_test), 1))
y_test = numpy.reshape(y_test.ravel(), (len(y_test), 1))

prediction = classifier.predict(x_test)

predicted_values = []
for i in prediction:
    if i == 0:
        predicted_values.append("Authorized")
    else:
        predicted_values.append("Forged")

actual_values = []
for i in y_test.ravel():
    if i == 0:
        actual_values.append("Authorized")
    else:
        actual_values.append("Forged")

# %% [markdown]
# ## Confusion Matrix

# %%
import seaborn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

labels = ["Forged", "Authorized"]

conf_matrix = confusion_matrix(actual_values, predicted_values, labels)

ax = plt.subplot()
seaborn.heatmap(conf_matrix, annot=True, ax=ax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

# %% [markdown]
# ## Accuracy 
# %%
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, prediction)
print(f"Accuracy Score: {accuracy}")
