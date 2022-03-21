from dataclasses import dataclass
from distutils.command.build_scripts import first_line_re
import re

######
# input_str = '''This room is called the green house. In it, you see: 
# 	a shovel
# 	the agent
# 	a substance called air
# 	a blue box (containing nothing)
# 	a jug (containing nothing)
# 	a sink, which is turned off. In the sink is: nothing.
# 	a bee hive. The bee hive door is closed. 
# You also see:
# 	A door to the hallway (that is closed)
# 	A door to the outside (that is open)
# '''
######

def parse_observation(obs):
    rules = []
    lines = obs.strip().split('\n') # remove the line (look around)
    first_line_match = re.match(r"(.*) is called the (.*)\. .*:", lines[0].strip())
    if first_line_match is None:
        return rules
    room = first_line_match.group(2)
    idx = 1
    while lines[idx].strip() != "You also see:":
        _, line_rules = parse_object_line(lines[idx], room, 'contains')
        rules.extend(line_rules)
        idx +=1
    
    for line in lines[idx+1: ]:
        door_match = re.match(r"A door to the (.*) \(that is (.*)\)", line.strip())
        if door_match is None:
            continue
        door_dest = door_match.group(1)
        door_status = door_match.group(2)
        rule = (room, f"door that is {door_status}", door_dest)
        rules.append(rule)
    return rules

def parse_inventory(inventory):
    rules = []
    lines = inventory.strip().split('\n')[1:]
    for line in lines:
        _, line_rules = parse_object_line(line, 'agent', 'has')
        rules.extend(line_rules)
    return rules

def parse_object(obj):
    if obj == 'the agent':
        return 'agent'
    elif obj.startswith("a substance called"):
        return ' '.join(obj.split()[3:])
    else:
        return obj[2:] # remove 'a '

def parse_object_with_property(obj_pro, subject, relation):
    templates = [r"(.*), which is (.*)",
                 r"(.*) \(containing (.*)\)",
                 r"(.*), currently reading a temperature of (.*)"]
    obj = None
    rules = []
    for n, template in enumerate(templates):
        match = re.match(template, obj_pro)
        if match is not None:
            obj = parse_object(match.group(1))
            
            if n == 0:
                rel = 'is'
            elif n == 1:
                rel = 'contains'
            else:
                rel = 'reads'
            
            targets = match.group(2)
            rules.append((subject, relation, obj))

            if n == 1:
                targets = split_obj_list(targets)
                for target in targets:
                    _, rule_target = parse_object_line(target, obj, rel)
                    rules.extend(rule_target)
            else:
                rules.append((obj, rel, targets))
            break

    return obj, rules

def parse_object_line(obj_line, subject, relation):
    rules = []
    obj_line = obj_line.strip().split('.')
    if ',' in obj_line[0] or '(' in obj_line[0]:
        obj, obj_rules = parse_object_with_property(obj_line[0], subject, relation)
        rules.extend(obj_rules)
    else:
        obj = parse_object(obj_line[0])
        rules.append((subject, relation, obj))
    if len(obj_line) > 1:
        obj_property = obj_line[1].strip()
        obj_property_templates = [r"(.*) the (.*) is: (.*)", r"The (.*) door is (.*)"]
        for n, obj_property_template in enumerate(obj_property_templates):
            match = re.match(obj_property_template, obj_property)
            if match is not None:
                if n == 0:
                    obj_relation = match.group(1).lower()
                    targets = match.group(3)
                    if targets != 'nothing':
                        targets = split_obj_list(targets)
                        for target in targets:
                            _, obj_rules = parse_object_line(target, obj, obj_relation)
                            rules.extend(obj_rules)
                elif n == 1:
                    target = match.group(2)
                    rules.append((obj, 'is', target))
    return obj, rules

def split_obj_list(obj_list):
    left_parenthese = 0
    objects = []
    curr_obj = ''
    for n, c in enumerate(obj_list):
        if c == ',' and left_parenthese == 0 \
            and n+6 < len(obj_list) and obj_list[n+2:n+7] != 'which':
            objects.append(curr_obj)
            curr_obj = ''
        else:
            if c == '(':
                left_parenthese += 1
            elif c == ')':
                left_parenthese -= 1
            curr_obj += c
    objects.append(curr_obj)
    return objects

# rules = parse_observation(input_str)
# print(input_str)
# for rule in rules:
#     print(rule)