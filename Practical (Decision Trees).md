#### Task 1
*Use the Tree data structure below; write code to build the tree from figure 1.2 in Daumé.*
   
```python
tree = Tree(data = "isSystems?", left = 'like', 
            right = Tree(data = "takenOtherSys?", left = Tree(data = 'morning?', left = 'like', right = 'nah'), 
            right = Tree(data = 'likedOtherSys?', left = 'nah', right = 'like')) )
```
                                                       
Output:

```python
Tree('isSystems?') { left = 'like', 
                     right = Tree('takenOtherSys?') { left = Tree('morning?') { left = 'like', right = 'nah' }, 
                     right = Tree('likedOtherSys?') { left = 'nah', right = 'like' } } } 
```
  
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
  file.write(c)
  
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
