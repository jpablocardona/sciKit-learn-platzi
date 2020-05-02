import pandas as pd

from sklearn.cluster import MeanShift

""" No supervisado meanshift """
if __name__ == "__main__":
    dataset = pd.read_csv("./data/candy.csv")

    x = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(x)
    print(max(meanshift.labels_))
    print("=" * 64)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_
    print("=" * 64)
    print(dataset)
