import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64

def plot_confusion_matrix(cm, class_names=None):
    if class_names is None:
        class_names = np.arange(cm.shape[0])  
        
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode('utf-8')


    return img_str
