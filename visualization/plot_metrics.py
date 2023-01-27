import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import torch
from sklearn import metrics

torch.manual_seed(0)

def plot_loss(epochs_list, loss, legend_name, path, img_name):

    plt.plot(epochs_list, loss, 'r--')
    plt.legend([legend_name])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(path, img_name))
    plt.close()

def plot_acc(epochs_list, acc, legend_name, path, img_name):

    plt.plot(epochs_list, acc, 'b--')
    plt.legend([legend_name])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(path, img_name))
    plt.close()

def tsne_plot(y, path, name, img_data = None, ehr_data = None):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    if img_data is not None:
        z = tsne.fit_transform(img_data)
        df1 = pd.DataFrame()
        df1["imgy"] = y
        df1["comp-1"] = z[:,0]
        df1["comp-2"] = z[:,1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df1.imgy.tolist(),
                    palette=sns.color_palette("hsv", 2),
                    data=df1).set(title="EMR_separation")


    if ehr_data is not None:
        z1 = tsne.fit_transform(ehr_data)
        df2 = pd.DataFrame()
        df2["ehry"] = y
        df2["comp-1"] = z1[:,0]
        df2["comp-2"] = z1[:,1]

    
        sns.scatterplot(x="comp-1", y="comp-2", hue=df2.ehry.tolist(),
                    palette=sns.color_palette("hot", 2),
                    data=df2).set(title="EMR_separation")
    #.figure.savefig("output.png")

    #fig = plot.get_figure()
    plt.savefig(os.path.join(path, name))
    plt.close()

def confusion_matrix_plot(actual, predicted, path, name):

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot(cmap='OrRd', xticks_rotation=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)


    # Giving name to the plot
    plt.title('Confusion Matrix', fontsize=24)

    plt.savefig(os.path.join(path, name), transparent=True, dpi=500)
    plt.close()


