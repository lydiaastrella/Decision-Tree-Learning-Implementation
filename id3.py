from sklearn import datasets
from copy import copy, deepcopy
import pandas as pd
import numpy as np
import math
import copy

#load dataset iris
iris = datasets.load_iris()

#convert scikit learn dataset to pandas dataframe
df_iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])
df_spek = pd.read_csv('iris.csv')
df_tennis = pd.read_csv('tennis.csv')

separator_tennis = round((4/5)*len(df_tennis.index))
training_tennis = (df_tennis.iloc[:separator_tennis, :]).reset_index(drop = True)
validation_tennis = (df_tennis.iloc[separator_tennis:, :]).reset_index(drop = True)
print(validation_tennis)

separator_iris = round((4/5)*len(df_iris.index))
training_iris = df_iris.iloc[:, :separator_iris]
validation_iris = df_iris.iloc[:, separator_iris:]

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
    if(split_in_info==0):
        gain_ratio_value = 999
    else:
        gain_ratio_value = gain / split_in_info
    return gain_ratio_value

#-------------CONTINUES HANDLING-------------------
def is_number (atribut): #apakah atribut tertentu berisi data number atau bukan
    try:
        a = atribut[0]/2
    except TypeError:
        return False
    return True

def count_unique_values (atribut):
    unique_atribut = set(atribut)
    unique_atribut_count = len(unique_atribut)
    return unique_atribut_count

def is_continuous(atribut): #apakah atribut tertentu berisi data continuous atau bukan
    if (is_number(atribut) and (count_unique_values(atribut) > 2)):
        return True
    else:
        return False

def continuous_attributes(data): #me-list index2 atribut yang datanya continuous
    tabel_continuous_attributes = []
    for idx_atribut in range (data.column-1):
        if(is_continuous(data.data_values[:,idx_atribut])):
            tabel_continuous_attributes.append(idx_atribut)
    return tabel_continuous_attributes

def sort_by_atribute(data, idx_atribut): #sorting data values supaya jadi sort berdasarkan atribut tertentu
    data.data_values = data.data_values[data.data_values[:,1].argsort()]

def get_hasil(data): #mendapat hasil data (yes/no)
    return data.data_values[:,data.column-1] 

def search_label_change(data): #cari column yang yes/no-nya berubah
    tabel_idx_label_change = []
    label_before = get_hasil(data)[0]
    for idx in range (data.row):
        label_now = get_hasil(data)[idx]
        if (label_before != label_now):
            tabel_idx_label_change.append(idx)
            label_before = label_now
    return tabel_idx_label_change

def create_changed_data (data, idx_atribut, idx_changed_label): #menghasilkan data baru yang sudah diganti atributnya (splitting)
    data_temp = copy.deepcopy(data)
    change_continuous_values(data_temp, idx_atribut, idx_changed_label)
    return data_temp

def idx_best_gain (data, idx_atribut, tabel_idx_label_change):
    max_gain = gain_info(data, idx_atribut)

    idx_best = tabel_idx_label_change[0]

    for idx_changed_label in tabel_idx_label_change:
        data_temp = copy.deepcopy(data)
        change_continuous_values(data_temp, idx_atribut, idx_changed_label)
        if (gain_info(data_temp, idx_atribut) > max_gain):
            max_gain = gain_info(data_temp, idx_atribut)
            idx_best = idx_changed_label

    return idx_best

def change_continuous_values (data, idx_atribut, idx_changed_label): 
    new_value = str(data.data_values[idx_changed_label,idx_atribut])
    for idx in range (data.row):
        if (idx < idx_changed_label):
            data.data_values[idx,idx_atribut] = "< " + new_value
        else:
            data.data_values[idx,idx_atribut] = ">= " + new_value
    
    data.data_properties=[]
    for i in range (data.column):
        column={}
        for x in data.data_values[:, i]:
            if(not(x in column)):
                count=0
                for y in data.data_values[:, i]:
                    if ( y == x):
                        count+=1
                value = ValueProperties(x,count)
                for z in data.target_attribute:
                    count_target=0
                    for j in range (data.row):
                        if(data.data_values[j][data.column-1]==z and data.data_values[j][i]==x):
                            count_target+=1
                    value.add_label(z,count_target)
                column[x]=value
        data.data_properties.append(column)

def change_continuous_atributes (data):
    for idx_atribut in continuous_attributes(data):
        sort_by_atribute(data, idx_atribut)
        tabel_idx_label_change = search_label_change(data)
        idx_split = idx_best_gain(data, idx_atribut, tabel_idx_label_change)
        change_continuous_values(data, idx_atribut, idx_split)

#-------------HANDLE MISSING VALUE------------
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
    
def most_common_target_value(data, attr, row): #for missing attribute value issues
    count = 0
    name = ""
    target = data.data_values[row][len(data.data_values[row])-1]
    iterate_data = iter(data.data_properties[attr])
    for i in range(len(data.data_properties[attr])):
        a = next(iterate_data)
        if (count < (data.data_properties[attr][a].label[target])):
            count = data.data_properties[attr][a].label[target]
            name = data.data_properties[attr][a].name
    return [name,count]

def change_missing_value (data):
    need_to_be_deleted = []
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
                    data.data_properties[j][''].label[data.target_attribute[i]] -= 1
                    if not(j in need_to_be_deleted):
                        need_to_be_deleted.append(j)
    for x in need_to_be_deleted:
        del data.data_properties[x]['']
    return data

#-------------ALGO------------
def id3(data, dataframe):
    print("--------------ID3------------")
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

def c45(data, dataframe):
    #handle missing value attributes
    print("--------------C45------------")
    data = change_missing_value(data)

    try:
        change_continuous_atributes(data)
    except(IndexError):
        pass

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
        #count gain ratio for every attributes
        gain = gain_ratio(data, 0)
        best_attr = 0
        for x in range(len(data.attributes)-1):
            if(gain < gain_ratio(data, x+1)):
                gain = gain_ratio(data, x+1)
                best_attr = x+1
        root = Vertex(data.attributes[best_attr])
        if(len(data.data_properties[best_attr]) > 0):
            
            #check every possible value of best_attr
            iterate_data = iter(data.data_properties[best_attr])
            for i in range(len(data.data_properties[best_attr])):
                attr_value = next(iterate_data)
                new_subset_data, df = get_data_certain_value(df, data.attributes[best_attr], attr_value)
                subtree = c45(new_subset_data, df)
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

def prune(tree, validation_data, validation_dataframe, rules, isContinuous):
    # PRUNING FUNCTIONS
    def make_rules(tree, rule):
        global rules
        if(len(tree.edges) > 0):
            for i in range(len(tree.edges)):
                newrule = rule.copy()
                newrule[tree.name] = tree.edges[i].name
                make_rules(tree.edges[i].target_vertex,newrule)
        else:
            rule["result"] = tree.name
            rules.append(rule)

    def rules_to_tree(rules):
        nodes = {}
        for rule in rules:
            for key in rule.keys():
                if key != 'result':
                    if key not in nodes:
                        nodes[key] = set([])
                    nodes[key].add(rule[key])
                else:
                    if rule[key] not in nodes:
                        nodes[rule[key]] = {}

        vertices = []
        for node in list(nodes.keys()):
            vertex = Vertex(node)
            vertices.append(vertex)

        for node in list(nodes.keys()):
            for value in nodes[node]:
                found = False
                rule_idx = 0
                while not found:
                    if (node in rules[rule_idx]):
                        if (str(rules[rule_idx][node]) == str(value)):
                            found = True
                            keys_in_rule = list(rules[rule_idx].keys())
                            next_node = keys_in_rule[keys_in_rule.index(node)+1]
                            if (next_node != 'result'):
                                edge = Edge(value, vertices[list(nodes.keys()).index(next_node)])
                                vertices[list(nodes.keys()).index(node)].add_edge(edge)
                            else:
                                edge = Edge(value, vertices[list(nodes.keys()).index(rules[rule_idx]['result'])])
                                vertices[list(nodes.keys()).index(node)].add_edge(edge)
                    rule_idx += 1
        return vertices[0]

    def pruning(validation_data, validation_dataframe, rules, isContinuous):
        best_ruleset = rules
        best_error = count_errors(predict_results(validation_data, rules), validation_data.data_values[:, validation_data.column-1], isContinuous)
        for rule in rules:
            pruning_dataframe = deepcopy(validation_dataframe)
            pruning_data = deepcopy(validation_data)
            keys = list(rule.keys())
            init_n_keys = len(keys)
            for i in range(init_n_keys-1):
                temp_rules = deepcopy(rules)
                for rule2 in temp_rules:
                    if (rule == rule2):
                        temp_rule = rule2
                for j in range(len(list(temp_rule.keys()))-1, i, -1):
                    if (list(temp_rule.keys())[i]) in temp_rule:
                        del temp_rule[list(temp_rule.keys())[j]]
                if (len(list(temp_rule.keys())) != 0):
                    iteration_data = deepcopy(pruning_data)
                    iteration_dataframe = deepcopy(pruning_dataframe)
                    for key in list(temp_rule.keys()):
                        if (key != 'result'):
                            iteration_data, iteration_dataframe = get_data_certain_value(iteration_dataframe, key, temp_rule[key])
                    temp_rule['result'] = most_common_value(pruning_data, pruning_data.column-1)[0]
                prediction = predict_results(validation_data, temp_rules)
                errors = count_errors(prediction, validation_data.data_values[:, validation_data.column-1], isContinuous)
                if (errors < best_error):
                    best_error = errors
                    best_ruleset = temp_rules
        return best_ruleset

    # predict data results using rules
    def predict_results(data, rules):
        results = []
        for datum in data.data_values:
            for rule in rules:
                keys = list(rule.keys())
                values = list(rule.values())
                unmatched = False
                for i in range(len(keys)-1):
                    if (keys[i] in data.attributes):
                        if (datum[data.attributes.index(keys[i])] != values[i]):
                            unmatched = True
                if not unmatched:
                    results.append(rule['result'])
        return results

    # return number of errors (discreet only for now)
    def count_errors(r1, r2, isContinuous):
        if len(r1) != len(r2):
            return 999
        else:
            cummulative_error = 0
            if isContinuous:
                for i in range(len(r1)):
                    cummulative_error += (r1[i] - r2[i])*(r1[i] - r2[i])/2
            else:
                for i in range(len(r1)):
                    if (r1[i] != r2[i]):
                        cummulative_error += 1
            return cummulative_error

    make_rules(tree, {})
    new_rules = pruning(validation_data, validation_dataframe, rules, isContinuous)
    final = rules_to_tree(new_rules)
    return final

def print_tree(tree,idx):
    print(tree.name)
    if(len(tree.edges) > 0):
        for i in range(len(tree.edges)):
            if(idx > 0):
                for a in range(idx):
                    print("    ", end ='')
            print("++++ (", tree.edges[i].name,")", end = ' ')
            print_tree(tree.edges[i].target_vertex,idx+1)

rules = []

data_training_tennis = Data(training_tennis)
data_training_iris = Data(training_iris)

hasil_id3 = id3(data_training_iris, training_iris)
print_tree(hasil_id3,0)

print(data_training_iris.data_values)

hasil_c45 = c45(data_training_iris, training_iris)
print_tree(hasil_c45,0)

data_validasi_tennis = Data(validation_tennis)
data_validasi_iris = Data(validation_iris)
hasil_diprune = prune(hasil_c45, data_validasi_iris, validation_iris, rules, True)
print_tree(hasil_diprune, 0)
