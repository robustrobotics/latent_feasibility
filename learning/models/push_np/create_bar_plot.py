import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_probabilities(probabilities, args):

    sns.set_theme(style="whitegrid")
    # Load probabilities into a pandas DataFrame
    df = pd.DataFrame(probabilities, columns=["Probability"])

    # Create bar plot
    sns.displot(df, x="Probability")

    # Save the plot with the instance name
    plot_path = os.path.join("learning", "data", "pushing", args.file_name, args.instance_name, "probabilities_plot.png")
    plt.savefig(plot_path)
    plt.close()