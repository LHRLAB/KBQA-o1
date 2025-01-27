"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import pickle
import json
import os
import shutil
import re
from typing import List
from utils.executor.sparql_executor import get_label_with_odbc


def dump_to_bin(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load_bin(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)


def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def mkdir_f(prefix):
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    os.makedirs(prefix)


def mkdir_p(prefix):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

def extract_mentioned_entities_from_sexpr(expr:str) -> List[str]:
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    entitiy_tokens = []
    for t in toks:
        # normalize entity
        if t.startswith('m.') or t.startswith('g.'):
            entitiy_tokens.append(t)
    return entitiy_tokens

def extract_mentioned_entities_from_sparql(sparql:str) -> List[str]:
    """extract entity from sparql"""
    sparql = sparql.replace('(',' ( ').replace(')',' ) ')
    toks = sparql.split(' ')
    toks = [x.replace('\t.','') for x in toks if len(x)]
    entity_tokens = []
    for t in toks:
        if t.startswith('ns:m.') or t.startswith('ns:g.'):
            entity_tokens.append(t[3:])
        
    entity_tokens = list(set(entity_tokens))
    return entity_tokens

def extract_mentioned_relations_from_sparql(sparql:str):
    """extract relation from sparql"""
    sparql = sparql.replace('(',' ( ').replace(')',' ) ')
    toks = sparql.split(' ')
    toks = [x for x in toks if len(x)]
    relation_tokens = []
    for t in toks:
        if (re.match("ns:[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t.strip()) 
            or re.match("ns:[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t.strip())):
            relation_tokens.append(t[3:])
    
    relation_tokens = list(set(relation_tokens))
    return relation_tokens


def extract_mentioned_relations_from_sexpr(sexpr:str)->List[str]:
    sexpr = sexpr.replace('(',' ( ').replace(')',' ) ')
    toks = sexpr.split(' ')
    toks = [x for x in toks if len(x)]
    relation_tokens = []

    for t in toks:
        if (re.match("[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-z_]*",t.strip()) 
            or re.match("[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*",t.strip())):
            relation_tokens.append(t)
    relation_tokens = list(set(relation_tokens))
    return relation_tokens

def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r: # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r: # replace "." with " , "
        r = r.replace(".", " , ")
    return r