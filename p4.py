import pandas as pd
from pprint import pprint
from sklearn.feature_selection import mutual_info_classif
from collections import Counter

def id3(df, target_attribute, attribute_names, default_class=None):
    cnt=Counter(x for x in df[target_attribute])
    if len(cnt)==1:
        return next(iter(cnt))
    
    elif df.empty or (not attribute_names):
         return default_class

    else:
        gainz = mutual_info_classif(df[attribute_names],df[target_attribute],discrete_features=True)
        index_of_max=gainz.tolist().index(max(gainz))
        best_attr=attribute_names[index_of_max]
        tree={best_attr:{}}
        remaining_attribute_names=[i for i in attribute_names if i!=best_attr]
        
        for attr_val, data_subset in df.groupby(best_attr):
            subtree=id3(data_subset, target_attribute, remaining_attribute_names,default_class)
            tree[best_attr][attr_val]=subtree
        
        return tree
    

df=pd.read_csv("p-tennis.csv")

attribute_names=df.columns.tolist()
print("List of attribut name")

attribute_names.remove("PlayTennis")

for colname in df.select_dtypes("object"):
    df[colname], _ = df[colname].factorize()
    
print(df)

tree= id3(df,"PlayTennis", attribute_names)
print("The tree structure")
pprint(tree)
List of attribut name
    Outlook  Temperature  Humidity  Windy  PlayTennis
0         0            0         0  False           0
1         0            0         0   True           0
2         1            0         0  False           1
3         2            1         0  False           1
4         2            2         1  False           1
5         2            2         1   True           0
6         1            2         1   True           1
7         0            1         0  False           0
8         0            2         1  False           1
9         2            1         1  False           1
10        0            1         1   True           1
11        1            1         0   True           1
12        1            0         1  False           1
13        2            1         0   True           0
The tree structure
{'Outlook': {0: {'Humidity': {0: 0, 1: 1}},
             1: 1,
             2: {'Windy': {False: 1, True: 0}}}}
