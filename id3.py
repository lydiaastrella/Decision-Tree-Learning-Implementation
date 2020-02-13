from sklearn import datasets
import pandas as pd
import numpy as np
import math

#load dataset iris
iris = datasets.load_iris()

#convert scikit learn dataset to pandas dataframe
df_iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])

df_tennis = pd.read_csv('tennis.csv')
df_tennis

#----------TREE----------
class Vertex:
    def __init__(self, name):
        self.name = name
        self.edges = []

    def add_edge(self, obj):
        self.edges.append(obj)

    def count_edge(self):
        return len(self.edges)

class Edge:
    def __init__(self, name, target_vertex):
        self.name = name
        self.target_vertex = target_vertex

#----------DATA----------
class ValueProperties:
    #value of attribute
    def __init__(self, value_name, count):
        self.name = value_name
        self.count = count
        self.label = {}
    
    #add each label's count that corresponds to attribute value
    def add_label(self, label_name, label_count):
        self.label[label_name] = label_count

class Data:
    #convert pandas dataframe to Data
    def __init__(self, dataframe):
        dataframe = dataframe.replace(np.nan, '', regex=True)
        self.row = len(dataframe)
        self.column = len(dataframe.columns)

        #list of attributes
        self.attributes = dataframe.columns

        #list of target labels
        self.target_attribute = []
        for x in dataframe[self.attributes[self.column-1]]:
            if(not(x in self.target_attribute)):
                self.target_attribute.append(x)
        self.target_attribute_count = len(self.target_attribute)

        self.attributes = self.attributes.tolist()
        del self.attributes[self.column-1]
        
        #2d list of values
        self.data_values = np.array(dataframe)

        #each value's properties
        self.data_properties=[]
        for i in range (self.column):
            column={}
            for x in self.data_values[:, i]:
                if(not(x in column)):
                    count=0
                    for y in self.data_values[:, i]:
                        if ( y == x):
                            count+=1
                    value = ValueProperties(x,count)
                    for z in self.target_attribute:
                        count_target=0
                        for j in range (self.row):
                            if(self.data_values[j][self.column-1]==z and self.data_values[j][i]==x):
                                count_target+=1
                        value.add_label(z,count_target)
                    column[x]=value
            self.data_properties.append(column)

# entropy global and specific entropy
def get_attribute_entropy(data, int_column, value_name):
    entropy=0
    value_property = data.data_properties[int_column][value_name]
    for x in data.target_attribute:
        if(value_property.label[x] > 0):
            probability = value_property.label[x] / value_property.count
            entropy -= probability * math.log(probability,2)
    return entropy

def get_global_entropy(data):
    entropy=0
    for x in data.target_attribute:
        value_property = data.data_properties[data.column-1][x]
        probability = value_property.count / data.row
        entropy -= probability * math.log(probability,2)
    return entropy

def get_data_certain_value (dataframe, attribute_name, value_name):
    new_dataframe = dataframe.copy()
    for i in range (len(new_dataframe)):
        if(new_dataframe[attribute_name][i] != value_name):
            new_dataframe = new_dataframe.drop(i)
    new_dataframe = new_dataframe.drop(attribute_name, axis=1)
    new_dataframe = new_dataframe.reset_index(drop = True)
    new_data = Data(new_dataframe)
    return new_data, new_dataframe

def gain_info(data, int_column):
    entropy = get_global_entropy(data)
    data_iter = iter(data.data_properties[int_column])
    for i in range(len(data.data_properties[int_column])):
        attr_val = next(data_iter)
        probability = data.data_properties[int_column][attr_val].count / (data.row)
        entropy -= (probability) * (get_attribute_entropy(data, int_column, attr_val))
    gain = entropy
    return gain

def gain_ratio(data, int_column):
    split_in_info=0
    iterate_data = iter(data.data_properties[int_column])
    for i in range(len(data.data_properties[int_column])):
        value_name = next(iterate_data)
        value_property = data.data_properties[int_column][value_name]
        probability = value_property.count / data.row
        split_in_info -= probability * math.log(probability,2)
        
    gain = gain_info(data,int_column)
    gain_ratio_value = gain / split_in_info
    return gain_ratio_value

def most_common_value(data, col):
    count = 0
    name = ""
    iterate_data = iter(data.data_properties[col])
    for i in range(len(data.data_properties[col])):
        a = next(iterate_data)
        if(count <= data.data_properties[col][a].count):
            count = data.data_properties[col][a].count
            name = data.data_properties[col][a].name
    return [name,count]

def change_missing_value (data):
    for j in range (data.column) :
        for i in range (data.row):
            if data.data_values[i,j] == '':
                data.data_values[i,j] = most_common_value(data, j)[0]
                found = False
                i=0
                while (not(found) and i<data.target_attribute_count):
                    if(data.data_values[i,data.column-1]==data.target_attribute[i]):
                        found = True
                    else:
                        i+=1
                if (found):
                    data.data_properties[j][data.data_values[i,j]].label[data.target_attribute[i]] += 1
                del data.data_properties[j]['']
    return data

def id3(data, dataframe):
    df = dataframe
    #if all examples are positive or negative
    if(data.column > 1):
        if (most_common_value(data, data.column-1)[1] == (data.row)):
            root = Vertex(most_common_value(data,data.column-1)[0])
            return root    
    
    #if attributes is empty
    if(data.column-1 == 0):
        val = most_common_value(data,0)
        root = Vertex(val[0])
        return root
    else:
        #count gain information for every attributes
        gain = gain_info(data, 0)
        best_attr = 0
        for x in range(len(data.attributes)-1):
            if(gain < gain_info(data, x+1)):
                gain = gain_info(data, x+1)
                best_attr = x+1
        root = Vertex(data.attributes[best_attr])
        if(len(data.data_properties[best_attr]) > 0):
            
            #check every possible value of best_attr
            iterate_data = iter(data.data_properties[best_attr])
            for i in range(len(data.data_properties[best_attr])):
                attr_value = next(iterate_data)
                new_subset_data, df = get_data_certain_value(df, data.attributes[best_attr], attr_value)
                subtree = id3(new_subset_data, df)
                new_edge = Edge(attr_value, subtree)
                root.add_edge(new_edge)
                df = dataframe
        else:
            iterate_data = iter(data.data_properties[best_attr])
            attr_value = next(iterate_data)
            val = Vertex(most_common_value(data, data.column-1)[0])
            new_edge = Edge(attr_value, val)
            root.add_edge(new_edge)
    return root
            
def print_tree(tree,idx):
    print(tree.name)
    if(len(tree.edges) > 0):
        for i in range(len(tree.edges)):
            if(idx > 0):
                for a in range(idx):
                    print("    ", end ='')
            print("++++ (", tree.edges[i].name,")", end = ' ')
            print_tree(tree.edges[i].target_vertex,idx+1)

data_tennis = Data(df_tennis)
hasil = id3(data_tennis, df_tennis)
print_tree(hasil,0)