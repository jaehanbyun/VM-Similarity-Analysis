import json
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import jaccard, hamming
import numpy as np

# Flatten nested JSON metadata
def flatten_json(json_obj, prefix=''):
    flattened = {}
    for key, value in json_obj.items():
        if isinstance(value, dict):
            flattened.update(flatten_json(value, prefix=prefix + key + '_'))
        else:
            flattened[prefix + key] = value
    return flattened

def levenshtein_distance(string_a, string_b):
    m, n = len(string_a), len(string_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if string_a[i - 1] == string_b[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]

def levenshtein_similarity(string_a, string_b):
    distance = levenshtein_distance(string_a, string_b)
    max_length = max(len(string_a), len(string_b))
    similarity = 1 - (distance / max_length)
    return similarity
def jaccard_similarity_categorical(cat_a, cat_b):
    set_a = set([cat_a])
    set_b = set([cat_b])
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union)

def numerical_similarity(num_a, num_b):
    dist = euclidean_distances([[num_a]], [[num_b]])
    return 1 / (1 + dist[0][0])

def weighted_similarity(feature_a, feature_b, weight, feature_type):
    if feature_type == 'categorical':
        similarity = jaccard_similarity_categorical(feature_a, feature_b)
    elif feature_type == 'numerical':
        similarity = numerical_similarity(feature_a, feature_b)
    elif feature_type == 'levenshtein':
        similarity = levenshtein_similarity(feature_a, feature_b)
    else:
        raise ValueError('Invalid feature type')
    return weight * similarity

def cosine_similarity_one_hot(one_hot_a, one_hot_b, one_hot_index, category_len):
    cat_a = one_hot_a[:, one_hot_index:one_hot_index + category_len]
    cat_b = one_hot_b[:, one_hot_index:one_hot_index + category_len]
    return cosine_similarity(cat_a, cat_b)[0][0]

# MetaSim algorithm
def metasim(metadata_a, metadata_b, cat_feature_weights, num_feature_weights, hamming_feature_weights, vm_categories):
    # Flatten JSON objects
    flat_a = flatten_json(metadata_a['metadata'])
    flat_b = flatten_json(metadata_b['metadata'])

    # Convert dictionaries to pandas DataFrames
    df_a = pd.DataFrame([flat_a])
    df_b = pd.DataFrame([flat_b])

    combined_df = pd.concat([df_a, df_b], axis=0)

    # Update vm_categories based on the actual data
    for feature, _ in cat_feature_weights.items():
        unique_values_a = df_a[feature].unique()
        unique_values_b = df_b[feature].unique()
        unique_values = list(set(unique_values_a) | set(unique_values_b))
        vm_categories[feature] = unique_values
    print(vm_categories)

    # One-hot encode categorical features using the provided vm_categories dictionary
    enc = OneHotEncoder()
    for category, values in vm_categories.items():
        combined_df[category] = pd.Categorical(combined_df[category], categories=values)
    one_hot_encoded = enc.fit_transform(combined_df)
    one_hot_a = one_hot_encoded[0]
    one_hot_b = one_hot_encoded[1]

    # Calculate weighted similarity for each feature
    total_similarity = 0
    total_weight = sum(cat_feature_weights.values()) + sum(num_feature_weights.values()) + sum(extra_feature_weights.values())

    # Categorical features
    for feature, weight in cat_feature_weights.items():
        feature_sim = weighted_similarity(df_a[feature].item(), df_b[feature].item(), weight, 'categorical')
        total_similarity += feature_sim

    # Numerical features
    for feature, weight in num_feature_weights.items():
        feature_sim = weighted_similarity(float(df_a[feature]), float(df_b[feature]), weight, 'numerical')
        total_similarity += feature_sim

    # Extra distance features
    for feature, weight in extra_feature_weights.items():
        feature_sim = weighted_similarity(df_a[feature].astype(str), df_b[feature].astype(str), weight, 'levenshtein')
        total_similarity += feature_sim

    # Normalize the similarity score
    normalized_similarity = total_similarity / total_weight

    return normalized_similarity

# Predefined pool of categories and their possible values
vm_categories = {
    'cloud provider': ['Openstack', 'Cloudstack', 'AWS', 'GCP', 'Azure'],
    'OS_type': ['Ubuntu', 'CentOS', 'Debian', 'Windows'],
    'OS_version': ['20.04', '18.04', '10', '8'],
    'status': ['Running', 'Stopped', 'Paused', 'Suspended'],
    'region': ['RegionOne', 'RegionTwo']
}

cat_feature_weights = {
    'cloud provider': 0.2,
    'OS_type': 0.1,
    'OS_version': 0.1,
    'status': 0.1,
    'region': 0.2,
}

num_feature_weights = {
    'OS_version': 0.05,
    'flavor_vcpu': 0.1,
    'flavor_mem': 0.15,
    'volume_size': 0.15
}

extra_feature_weights = {
    'name': 0.1,
    'network_id': 0.1,
    'subnet_id': 0.1,
    'owner_name': 0.05
}

# Example VM metadata
with open('metadata_sets/metasim1.json') as m1:
    vm_metadata_a = json.load(m1)

with open('metadata_sets/metasim2.json') as m2:
    vm_metadata_b = json.load(m2)

with open('metadata_sets/metasim3.json') as m3:
    vm_metadata_c = json.load(m3)


similarity1 = metasim(vm_metadata_a, vm_metadata_b, cat_feature_weights, num_feature_weights, extra_feature_weights, vm_categories)
similarity2 = metasim(vm_metadata_a, vm_metadata_c, cat_feature_weights, num_feature_weights, extra_feature_weights, vm_categories)

print("Similarity a ~ b score:", similarity1)
print("Similarity a ~ c score:", similarity2)