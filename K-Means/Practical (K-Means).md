#### Task 1
*Write a clustering problem generator with signature.*
 
```python
def scatter_clusters(centers, spread, n_points):
  clusters = []
  x_c = []
  y_c = []
  for i in range(len(centers)):
    x = np.random.randint(low = centers[i][0] - spread[0], high = centers[i][0] + spread[0], size=n_points)
    y = np.random.randint(low = centers[i][1] - spread[1], high = centers[i][1] + spread[1], size=n_points)
    dots = [[x[w],y[w]] for w in range(len(x))]
    clusters.append(dots)    
  return clusters
  ```

*For k=3, generate easy and hard problems and plot them; the easy problem might look like figure 3.13 from Daumé*

To generate this graph see use the following part of the script.
```python
def classif(data):
	""" unclusters data """
	data = [item for sublist in data for item in sublist]
	return data

dots_easy = scatter_clusters([[1, 7], [13, 12], [25, 3]],[4, 4], 50)
dots_medium = scatter_clusters([[1, 7], [13, 12], [25, 3]], [7, 7], 50)
dots_hard = scatter_clusters([[1, 7], [13, 12], [25, 3]], [10, 12], 50)

def plot(data, color):
  x = []
  y = []
  for i in range(len(data)):
    x.append(data[i][0])
    y.append(data[i][1])
  _ = plt.scatter(x, y, color=color, alpha=0.5)

_ = plt.figure(figsize=[8, 8])
plt.subplot(2, 2, 1)
_ = plt.title('An example of an easy problem')
plot(classif(dots_easy), '#9ddca6')
plt.subplot(2, 2, 2)
_ = plt.title('An example of a medium problem')
plot(classif(dots_medium), '#eeb87c')
plt.subplot(2, 2, 3)
_ = plt.title('An example of a hard problem')
plot(classif(dots_hard), '#e99aee')
plt.tight_layout()
plt.show()
```

#### Task 2
*Implement K-means clustering as shown in Daumé:*
```python
def kmeans_cluster_centers(k, points, centers_guess = None, max_iterations = 100, tolerance = 1e-100):
  points = [item for sublist in points for item in sublist]
  x = np.asarray(points)
  guess = np.random.normal(loc = np.mean(x, axis=0), scale=np.std(x, axis=0), size=[k,2]) if centers_guess is None else centers_guess
  clusters = [[] for i in range(k)]
  tol = 1
  for y in range(max_iterations):
    if tol < tolerance:
      return guess, clusters
    old = np.copy(guess) 
    for i in range(len(x)):
      res = min(range(k), key=lambda c: np.linalg.norm(x[i] - guess[c]))
      clusters[res].append(points[i])       
    for nk in range(k):     
      guess[nk] = np.mean(clusters[nk], axis=0)
    tol = np.linalg.norm(old - guess)      
  return guess, clusters
```

We return both clustered data and centroids in order to be able to classify data in the future.

*Replot your problems at 5 stages (random initialisation, 25%, 50%, 75%, 100% of iterations), using colours to assign points to clusters.*

To generate this graph see use the following part of the script.

```python
def plot_cluster(data, k=3):
  data_res = data
  colors = ['#9ddca6', '#eeb87c', '#e99aee', 'green', 'cyan', 'red']
  for x in range(k):
    plot(data_res[x], color = colors[x])

_ = plt.figure(figsize=[8, 12])
plt.subplot(3, 2, 1)
_ = plt.title('K-Means with 100% iterations')
plot_cluster(kmeans_cluster_centers(3, dots_easy, max_iterations=100)[1])
plt.subplot(3, 2, 2)
_ = plt.title('K-Means with 75% iterations')
plot_cluster(kmeans_cluster_centers(3, dots_easy, max_iterations=75)[1])
plt.subplot(3, 2, 3)
_ = plt.title('K-Means with 50% iterations')
plot_cluster(kmeans_cluster_centers(3, dots_easy, max_iterations=50)[1])
plt.subplot(3, 2, 4)
_ = plt.title('K-Means with 25% iterations')
plot_cluster(kmeans_cluster_centers(3, dots_easy, max_iterations=25)[1])
plt.subplot(3, 2, 5)
_ = plt.title('K-Means with random inialisation')
plot_cluster(kmeans_cluster_centers(3, dots_easy, max_iterations=np.random.randint(100))[1])
plt.tight_layout()
plt.show()
```

#### Task3
*Study the performance of these two implementations: memory, speed, quality; compare against scipy.cluster.vq.kmeans.*

 * memory scipy: 125.71 MiB
 * memory my algorithm: 127.89 MiB
 * speed scipy: CPU times: user 214 ms, sys: 33.5 ms, total: 248 ms
 * speed my algorithm: CPU times: user 1.27 s, sys: 29.2 ms, total: 1.3 s

To measure these parameters I wrote the following code:
```python
from scipy.cluster.vq import vq, kmeans, whiten

%%capture
!pip install memory_profiler
%load_ext memory_profiler

%%time
%memit
whitened = whiten(classif(dots_easy))
book = np.array((whitened[0],whitened[2]))  
codebook, distortion = kmeans(whitened, 3)
plt.scatter(whitened[:, 0], whitened[:, 1], c= '#9ddca6')
plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
plt.show()

%%time
%memit

plot_cluster(kmeans_cluster_centers(3, dots_easy)[1]) # index is 1, so that code returns only the clustered data
```

In terms or performance, visually the algorithms performed the same, yet judging by the other criteria scipy is significantly faster.


#### Task 4
*Compute the performance of your algorithm as percent of points assigned the correct cluster. Graph this as a function of iterations, at 10% intervals.*

To see this graph, consider the following part of the script
```python
def mean_sort(x):
  if len(x):
    return np.mean(x)
  else:
    return 0

def score(data1, data2):
  data_1 = sorted(data1, key=mean_sort)
  data_2 = sorted(data2, key=mean_sort)
  all = len(classif(data1))
  correct = 0
  for x in range(len(data2)):
    for i in range(len(data_2[x])):
      if np.all(data_2[x][i] in data_1[x]):
        correct += 1
  return correct/all
  
performance = []
for i in range(10, 101, 10):
  f1 = kmeans_cluster_centers(3, dots_easy, max_iterations=i)[1]
  res = score(dots_easy, f1)/100
  performance.append(res)
  
_ = plt.plot(performance, marker='o', linestyle='dashed', color = '#b82988')
_ = plt.title('Performance vs Number of Iterations')
_ = plt.xlabel('Number of Iterations', fontsize=12)
_ = plt.ylabel('Score', fontsize=12)
_ = plt.yticks(np.arange(0, 1.1, step=0.1))
_ = plt.xticks([i for i in range(0, 10)], [i for i in range(10, 101, 10)])
plt.show()
```

*Make a random 10-90 test-train split; now you train on 90% of the data, and evaluate on the other 10%. How does the performance graph change?*

The performance became significantly more stable. To check it consider the following part of the script
```python
def train_test(data, split, k, shuffle = True):
  train_split = []
  test_split = []
  b = int((len(classif(data))*split)/k)
  for i in range(len(data)):
    if shuffle:
      np.random.shuffle(data[i])    
    train_split.append(data[i][:b])
    test_split.append(data[i][b:])
  return train_split, test_split
  
train, test = train_test(dots_easy, 0.9, 3)

def classify(centroids, test, k):
  clusters = [[] for i in range(k)]
  test = classif(test)
  test_np = np.asarray(test)
  for i in range(len(test)):
      res = min(range(k), key=lambda c: np.linalg.norm(test_np[i] - centroids[c]))
      clusters[res].append(test[i])
  return clusters

centroids, clustered_train = kmeans_cluster_centers(3, train)
clustered_test = classify(centroids, test, 3)

_ = plt.figure(figsize=[8, 4])
plt.subplot(1, 2, 1)
plot_cluster(clustered_train)
_ = plt.title('Clustered Train')
plt.subplot(1, 2, 2)
plot_cluster(clustered_test)
_ = plt.title('Clustered Test')
plt.tight_layout()
plt.show()
  
performance_train = []
for i in range(10, 101, 10):
  centroids_1 = kmeans_cluster_centers(3, train, max_iterations=i)[0]
  clustered = classify(centroids_1, test, 3)
  res = score(test, clustered)
  performance_train.append(res)
  
_ = plt.plot(performance_train, marker='o', linestyle='dashed', color = '#b82988')
_ = plt.title('Performance vs Number of Iterations \n on the Test Data')
_ = plt.xlabel('Number of Iterations', fontsize=12)
_ = plt.ylabel('Score', fontsize=12)
_ = plt.yticks(np.arange(0, 1.5, step=0.1))
_ = plt.xticks([i for i in range(0, 10)], [i for i in range(10, 101, 10)])
plt.show()
```

#### Task 5 
*Instead of a pure 10-90 split, divide your data into 10 portions.*

*Perform cross-validation on your training data: plot the mean of these performances against percent-of-iterations.*

To see the results, consider the following part of the script

```
def crossval(data, splits, k, iter=100):
  cl = len(classif(data))/k
  num_points = int(cl/splits)
  indexes = [num_points*i for i in range(1, splits+1)]
  scores = []

  for y in range(splits):    
    train_split = []
    test_split = []
    for i in range(k):
      test_split.append(data[i][:indexes[y]])
      train_split.append(data[i][indexes[y]:])
    centroids = kmeans_cluster_centers(3, train_split, max_iterations=iter)[0]
    clustered_test = classify(centroids, test_split, 3)
    scores.append(score(test_split, clustered_test))

  return np.mean(scores), np.std(scores)
  
performance_crossval = []
performance_std = []
for i in range(10, 101, 10):
  performance_crossval.append(crossval(dots_easy, 10, 3, iter=i)[0])
  performance_std.append(crossval(dots_easy, 10, 3, iter=i)[1])
    
low = np.asarray(performance_crossval) - np.asarray(performance_std)/2
high = np.asarray(performance_crossval) + np.asarray(performance_std)/2  

_ = plt.figure(figsize=[10, 6])
_ = plt.fill_between([i for i in range(0, 10)], high, low, alpha=0.5, facecolor='#ffff00' )
_ = plt.plot(performance_crossval, marker='o', linestyle='dashed', color = '#f048ca', label = 'algorithm performance')
_ = plt.xticks([i for i in range(0, 10)], [i for i in range(10, 101, 10)])
_ = plt.title('Mean performance on Cross-validation\n vs\n Number of Iterations')
_ = plt.xlabel('Number of Iterations', fontsize=12)
_ = plt.ylabel('Score', fontsize=12)
_ = plt.xlim((-0.1, 9.1))
_ = plt.legend(labels = ['algorithm performance', 'max performance'])
plt.show()
```
