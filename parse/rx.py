import csv
import re 
import ahocorasick


def build_rx(): 
    english_concepts = []
    with open('chinese_censored_topics.csv', 'r') as file:
        reader = csv.reader(file)
        a_tree = ahocorasick.Automaton()
        for idx, row in enumerate(reader):
            a_tree.add_word(row[0], (idx, row[0]))
            english_concepts.append(row[0])

    a_tree.make_automaton()

    return a_tree

