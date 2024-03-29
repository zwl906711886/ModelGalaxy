import itertools
import os
import time
from pyspark import SparkContext
from pyspark import SparkConf


conf = SparkConf().set("spark.driver.maxResultSize", "20g").set("spark.driver.memory", "30g")
sc = SparkContext(master='local[*]', appName='test', conf=conf)
def parseLine(tree, tree_number):
    """This function responsibles of parse the lines according to ,

     Parameters
     ----------
     tree : str
         string which describes the instances' distribution in some tress
     tree_number : int

     Returns
     -------
     list
          list of pairs : <key =  tree number + leaf number , value = instance number>
     """

    parts = tree.split(',')
    ans = []
    for i in range(0, len(parts)):
        ans.append(((tree_number, parts[i]), i))

    return ans


def combinations(row):
    """ This function return all the possible options of combinations

        Parameters
        ----------
        row : str
            string which describes instances that felt in the same leaf
        tree_number : int

        Returns
        -------
        list
             list of edges (pairs of instances)
        """
    l = sorted(row[1])
    k = row[0][1]
    return [(v) for v in itertools.combinations(l, 2)]


def addToEdge(edge):
    """ This function cleans the edges from some symbols and adds some symbols that are required later.

        Parameters
        ----------
        edge : str
            string (instance1, instance2)

        Returns

        str
            instance1 instance2 0 |{}|
        """
    e = str(edge).replace(",", "")
    e = e.replace("(", "")
    e = e.replace(")", "")
    return e + " 0 |{}|"

def main(list_paths, datasets_names_list, output1, output2):
    """ This function computes the co- co-occurrence score for each pair of instacnes.
        Then , removes the pairs with the weakest co-occurence scores (10% )
        The others pairs, are written to two different files: output1, output2

        Parameters
        ----------
        list_paths : list
            list of the datasets' paths
        datasets_names_list: list
            The datasets's names, orderd like the list_paths
        output1: str
           Path to the first output file
        output2: str
           Path to the second output file

        """
    list_paths = list_paths
    list_names = datasets_names_list

    dataset_names_dic = {}
    for num in range(0, len(list_names)):
        dataset_names_dic[num] = list_names[num]

    counter_data_set = 0
    # sc = SparkContext.getOrCreate(SparkConf())

    for path in list_paths:
        print(path)
        dataset_name = dataset_names_dic[counter_data_set]
        counter_data_set = counter_data_set + 1

        # Load and parse data file into an RDD of LabeledPoint

        data = sc.textFile(path)
        temp = []
        counter = 1

        for i in data.collect():
            instances_number = len(i.split(","))  # Find the instances number
            temp.append((i, counter))  # Add line number for each line
            counter = counter + 1

        data = sc.parallelize(temp)

        a = data.flatMap(lambda tp: parseLine(tp[0], tp[1])).groupByKey().map(
            lambda x: (x[0], list(x[1]))).collect()  # return : <number of tree , leaf number value: list of instances>

        a = sc.parallelize(a)

        selected_edge = a.map(lambda x: combinations(x)).flatMap(lambda x: x).map(lambda x: (x, 1)).reduceByKey(
            lambda tp1, tp2: tp1 + tp2).sortBy(lambda a: -a[1])
        edges_amount_before_filter = len(selected_edge.collect())
        precentage = 0.9
        wish_number_of_edges = edges_amount_before_filter * precentage

        d = sc.parallelize(selected_edge.take(int(wish_number_of_edges)))
        d = d.map(lambda x: x[0])

        gw_begin = ""
        for i in range(0, instances_number):
            gw_begin = gw_begin + "|{}|" + "\n"  # Save according to
        edge_gw = d.map(lambda edge: addToEdge(edge))

        s_gw = str(edge_gw.collect())
        s_gw = s_gw.replace(" '", "")
        s_gw = s_gw.replace("'", "")
        s_gw = s_gw.replace(",", "\n")
        s_gw = s_gw.replace("[", "")
        s_gw = s_gw.replace("]", "")
        print("s_gw")

        s_with_count = "" + str(instances_number) + " " + str(
            len(d.collect())) + "\n"  # string that save edges list with the number of appearence

        s = str(d.collect())

        s = s.replace("),", "\n")
        s = s.replace(" (", "")
        s = s.replace(",", "")
        s = s.replace("[", "")
        s = s.replace("]", "")
        s = s.replace("(", "")
        s = s.replace(")", "")
        dir_name = dataset_name
        os.system("mkdir " + output1 + dir_name)
        file1 = open(output1 + dir_name + '/' + dataset_name + ".gw","w")
        file1.write(gw_begin + s_gw)
        file1.close()
        file2 = open(output2 + dataset_name + ".in", "w")
        file2.write(s_with_count + s)
        file2.close()
        # dbutils.fs.put(output1 + dataset_name + ".gw", gw_begin + s_gw)
        # dbutils.fs.put(output2 + dataset_name + ".in", s_with_count + s)
        del data

        # The output files are required later by the graphlet correlation distance method
        # The format of Output1:
        # LEDA graph format

        # The format of output2 (.in file):
        # Input file describes the network in a simple text format.
        # The first line contains two integers n and e - the number of nodes and edges.
        # The following e lines describe undirected edges with space-separated ids of their endpoints.
        # Node ids should be between 0 and n-1 (see example.in).

# list1 : list of the datasets' paths
# list2: the datasets's names, orderd like the list_paths
list1 = ['output12/']
output_path = 'dataset'
list2 = [output_path]
# output1:Path to the first output file(.gw file)
# output2: Path to the second output file(.in file)
output1 = 'output24/gw/'
output2 = 'output24/in/'
main(list1, list2, output1, output2)
