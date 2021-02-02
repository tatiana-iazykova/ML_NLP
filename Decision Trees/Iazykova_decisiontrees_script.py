import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Tree:
  '''Create a binary tree; keyword-only arguments `data`, `left`, `right`.

  Examples:
    l1 = Tree.leaf("leaf1")
    l2 = Tree.leaf("leaf2")
    tree = Tree(data="root", left=l1, right=Tree(right=l2))
  '''

  def leaf(data):
    '''Create a leaf tree
    '''
    return Tree(data=data)

  # pretty-print trees
  def __repr__(self):
    if self.is_leaf():
      return "Leaf(%r)" % self.data
    else:
      return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right) 

  # all arguments after `*` are *keyword-only*!
  def __init__(self, *, data = None, left = None, right = None):
    self.data = data
    self.left = left
    self.right = right

  def is_leaf(self):
    '''Check if this tree is a leaf tree
    '''
    return self.left == None and self.right == None

  def children(self):
    '''List of child subtrees
    '''
    return [x for x in [self.left, self.right] if x]

  def depth(self):
    '''Compute the depth of a tree
    A leaf is depth-1, and a child is one deeper than the parent.
    '''
    return max([x.depth() for x in self.children()], default=0) + 1

# write code to build the tree from figure 1.2 in Daumé.
tree = Tree(data = "isSystems?", left = 'like', right = Tree(data = "takenOtherSys?", \
                                                       left = Tree(data = 'morning?', left = 'like', right = 'nah'), \
                                                       right = Tree(data = 'likedOtherSys?', left = 'nah', right = 'like')) )
print(tree)
# In your python code, load the following dataset and add a boolean "ok" column,
# where "True" means the rating is non-negative and "False" means the rating is negative
c = '''rating,easy,ai,systems,theory,morning
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
  file.write(c)

df = pd.read_csv('corpus.txt')

df['ok'] = df.rating > 0

# Write a function which takes a feature and computes the performance 
# of the corresponding single-feature classifier:
# the analysis of best/worst features you can find in the .md file
def single_feature_score(data, goal, feature):
  yes = data[data[feature] == True][goal]
  no = data[data[feature] == False][goal]
  score = (np.sum(yes.value_counts().idxmax() == yes) + np.sum(no.value_counts().idxmax() == no))/len(data)
  return score

def best_feature(data, goal, features):
  # optional: avoid the lambda using `functools.partial`
  return max(features, key=lambda f: single_feature_score(data, goal, f))

# Implement the DecisionTreeTrain and DecisionTreeTest algorithms from Daumé, returning Trees. +
# Add an optional maxdepth parameter to DecisionTreeTrain, which limits the depth of the tree produced.
# Plot performance against maxdepth.
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

def DecisionTreeTest(tree, test_point):
  if tree.is_leaf():
    return tree.data
  else: 
    if test_point[tree.data] == False:
      return DecisionTreeTest(tree.left, test_point)
    else:
      return DecisionTreeTest(tree.right, test_point)

def score(tree, data, goal):
  answer = 0
  for i in range(len(data)):
    pred = DecisionTreeTest(tree, data.iloc[i])
    if pred == data[goal].iloc[i]:
      answer += 1
  return answer/len(data)

performance = []

for i in range(1, 8):
  performance.append(score(DecisionTreeTrain(df, goal='ok', \
  features = ['easy', 'ai', 'systems', 'theory', 'morning'], maxdepth=i), df, goal = 'ok'))

_ = plt.plot(performance, marker='o', linestyle='dashed', color = '#b82988')
_ = plt.xticks(np.arange(len(performance)), np.arange(1, len(performance)+1))
_ = plt.xlabel('Tree depth', fontsize=12)
_ = plt.ylabel('Score', fontsize=12)
_ = plt.yticks(np.arange(0, 1, step=0.1))
_ = plt.title('Effect of the tree depth on the score', fontsize=14)
plt.show()
