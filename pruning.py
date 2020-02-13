#----------PRUNING----------
# turn decision tree into rule base
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

def prune(validation_data, validation_dataframe, rules):
    best_ruleset = rules
    best_error = count_errors(predict_results(validation_data, rules), validation_data.data_values[:, validation_data.column-1])
    for rule in rules:
        keys = list(rule.keys())
        init_n_keys = len(keys)
        for i in range(init_n_keys-1):
            temp_rules = rules.copy()
            for rule2 in temp_rules:
                if (rule == rule2):
                    temp_rule = rule2.copy()
            for i in range(i, len(list(temp_rule.keys()))):
                del temp_rule[keys[i]]
            dataframe = validation_dataframe
            data = validation_data
            for key in list(temp_rule.keys()):
                if (key != 'result'):
                    data, dataframe = get_data_certain_value(dataframe, key, temp_rule[key])
            temp_rule['result'] = most_common_value(data, data.column-1)[0]
            prediction = predict_results(data, temp_rules)
            errors = count_errors(prediction, data.data_values[:, data.column-1])
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
def count_errors(r1, r2):
    if len(r1) != len(r2):
        return -1
    else:
        n_errors = 0
        for i in range(len(r1)):
            if (r1[i] != r2[i]):
                n_errors += 1
        return n_errors