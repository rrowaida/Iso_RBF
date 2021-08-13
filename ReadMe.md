# Isomap

Run the instruction as: 

```bash
python isomap.py arg1 arg2 arg3'
```

Here agr1 is the dataset tags. Here value 1 means datasets of chairs, value 2 means dataset of lamps, value 3 means dataset of tables. In this code arg2 is the reduced dimension. Usually the value is 2-5. Third argument arg3 represents the neighborhood cluster size. For arg3, 15 is a standard value. 

# RBF

```bash
python rbf.py arg1 arg2 arg3'
```

Here agr1 is the dataset tags. Here value 1 means datasets of chairs, value 2 means dataset of lamps, value 3 means dataset of tables. In this code arg2 represents the neighborhood cluster size. For arg3, 15 is a standard value. Third argument arg3 is the reduced dimension. I usud the value 2-5.

# RBF_2

```bash
python rbf_2.py arg1 arg2 arg3'
```

This one takes A_matrix directly then compute RBF. Argument set is same as RBF.

# chop

```bash
python chop.py arg1 arg2 arg3'
```

chop cuts the z vector in one forth. Argument set is same as RBF.
