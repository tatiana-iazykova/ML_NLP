#### Task 1
*Use the Tree data structure below; write code to build the tree from figure 1.2 in Daumé.*
   
`tree = Tree(data = "isSystems?", left = 'like', 
            right = Tree(data = "takenOtherSys?", left = Tree(data = 'morning?', left = 'like', right = 'nah'), 
            right = Tree(data = 'likedOtherSys?', left = 'nah', right = 'like')) )`
                                                       
> Output:
>
> `Tree('isSystems?') { left = 'like', 
                     right = Tree('takenOtherSys?') { left = Tree('morning?') { left = 'like', right = 'nah' }, 
                     right = Tree('likedOtherSys?') { left = 'nah', right = 'like' } } }`
  
#### Task 2
*In your python code, load the following dataset and add a boolean "ok" column, where "True" means the rating is non-negative and "False" means the rating is negative.*

```python 
data = '''rating,easy,ai,systems,theory,morning
 2,True,True,False,True,False
 2,True,True,False,True,False
 2,False,True,False,False,False
 2,False,False,False,True,False
 2,False,True,True,False,True
 1,True,True,False,False,False
 1,True,True,False,True,False
 1,False,True,False,True,False
 0,False,False,False,False,True
 0,True,False,False,True,True
 0,False,True,False,True,False
 0,True,True,True,True,True
-1,True,True,True,False,True
-1,False,False,True,True,False
-1,False,False,True,False,True
-1,True,False,True,False,True
-2,False,False,True,True,False
-2,False,True,True,False,True
-2,True,False,True,False,False
-2,True,False,True,False,True'''

with open('corpus.txt', 'w') as file:
  file.write(data)
  
df = pd.read_csv('corpus.txt')

df['ok'] = df.rating > 0 

```
**Result**:

|    |   rating | easy   | ai    | systems   | theory   | morning   | ok    |
|---:|---------:|:-------|:------|:----------|:---------|:----------|:------|
|  0 |        2 | True   | True  | False     | True     | False     | True  |
|  1 |        2 | True   | True  | False     | True     | False     | True  |
|  2 |        2 | False  | True  | False     | False    | False     | True  |
|  3 |        2 | False  | False | False     | True     | False     | True  |
|  4 |        2 | False  | True  | True      | False    | True      | True  |
|  5 |        1 | True   | True  | False     | False    | False     | True  |
|  6 |        1 | True   | True  | False     | True     | False     | True  |
|  7 |        1 | False  | True  | False     | True     | False     | True  |
|  8 |        0 | False  | False | False     | False    | True      | False |
|  9 |        0 | True   | False | False     | True     | True      | False |
| 10 |        0 | False  | True  | False     | True     | False     | False |
| 11 |        0 | True   | True  | True      | True     | True      | False |
| 12 |       -1 | True   | True  | True      | False    | True      | False |
| 13 |       -1 | False  | False | True      | True     | False     | False |
| 14 |       -1 | False  | False | True      | False    | True      | False |
| 15 |       -1 | True   | False | True      | False    | True      | False |
| 16 |       -2 | False  | False | True      | True     | False     | False |
| 17 |       -2 | False  | True  | True      | False    | True      | False |
| 18 |       -2 | True   | False | True      | False    | False     | False |
| 19 |       -2 | True   | False | True      | False    | True      | False |


#### Task 3
*Write a function which takes a feature and computes the performance of the corresponding single-feature classifier:*

```python
def single_feature_score(data, goal, feature):
  yes = data[data[feature] == True][goal]
  no = data[data[feature] == False][goal]
  score = (np.sum(yes.value_counts().idxmax() == yes) + np.sum(no.value_counts().idxmax() == no))/len(data)
  return score
  ```
*Which feature is best? Which feature is worst?*

1. The best feature is 'systems' with score 0.8
2. The worst features are 'easy' and 'theory', both with score 0.6

#### Task 4 
*Implement the `DecisionTreeTrain` and `DecisionTreeTest` algorithms from Daumé, returning Trees. (Note: our dataset and his are different; we won't get the same tree.)*

**Decision Tree Train**

```python
def DecisionTreeTrain(data, goal, features):
    guess = data[goal].value_counts().idxmax()
    if np.all(data[goal] == data[goal].iloc[0]):
      return Tree.leaf(guess)
    elif not features:
      return Tree.leaf(guess)
    else:
      f = best_feature(data, goal, features)
      no = data[data[f] == False]
      yes = data[data[f] == True]
      features.remove(f)
      left = DecisionTreeTrain(data = no, goal = goal, features = features, maxdepth=maxdepth)
      right = DecisionTreeTrain(data = yes, goal = goal, features = features, maxdepth=maxdepth) 
      return Tree(data = f, left=left, right=right)   
```

> Output for our dataset:
> 
> `Tree('systems') { left = Tree('morning') { left = Tree('easy') { left = Tree('ai') { left = Leaf(True), right = Tree('theory') { left = Leaf(True), right = Leaf(True) } }, right = Leaf(True) }, right = Leaf(False) }, right = Leaf(False) }`

**Decision Tree Test**

```python
def DecisionTreeTest(tree, test_point):
  if tree.is_leaf():
    return tree.data
  else: 
    if test_point[tree.data] == False:
      return DecisionTreeTest(tree.left, test_point)
    else:
      return DecisionTreeTest(tree.right, test_point)
```
*How does the performance compare to the single-feature classifiers?*

> The single-feature classifiers performs worse than the full algorithm with maximum score for single-feature classifiers being 0.8 and the score for DecisionTreeTrain being 0.9

#### Task 5 
*Add an optional maxdepth parameter to DecisionTreeTrain, which limits the depth of the tree produced. Plot performance against maxdepth.*

New DecisionTreeTrain

```python
def DecisionTreeTrain(data, goal, features, maxdepth = None):
  maxdepth = float('inf') if maxdepth is None else maxdepth
  depth = 0
  while maxdepth > depth:
    guess = data[goal].value_counts().idxmax()
    if np.all(data[goal] == data[goal].iloc[0]):
      return Tree.leaf(guess)
    elif not features:
      return Tree.leaf(guess)
    else:
      f = best_feature(data, goal, features)
      no = data[data[f] == False]
      yes = data[data[f] == True]
      features.remove(f)
      maxdepth -= 1
      left = DecisionTreeTrain(data = no, goal = goal, features = features, maxdepth=maxdepth)
      right = DecisionTreeTrain(data = yes, goal = goal, features = features, maxdepth=maxdepth)  
      return Tree(data = f, left=left, right=right) 
 ```
![Performance](https://github.com/tatiana-iazykova/ML_NLP/blob/main/maxdepth_vs_score.png?raw=true)
