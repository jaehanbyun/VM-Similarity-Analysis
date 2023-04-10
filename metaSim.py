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

def categorical_similarity(cat_a, cat_b):
    return 1 - jaccard([cat_a], [cat_b])

def numerical_similarity(num_a, num_b):
    dist = euclidean_distances([[num_a]], [[num_b]])
    return 1 / (1 + dist[0][0])

def hamming_similarity(feature_a, feature_b):
    distance = hamming(feature_a.to_numpy(), feature_b.to_numpy())
    similarity = 1 - distance
    return similarity

def weighted_similarity(feature_a, feature_b, weight, feature_type):
    if feature_type == 'categorical':
        similarity = categorical_similarity(feature_a, feature_b)
    elif feature_type == 'numerical':
        similarity = numerical_similarity(feature_a, feature_b)
    elif feature_type == 'hamming':
        similarity = hamming_similarity(feature_a, feature_b)
    else:
        raise ValueError('Invalid feature type')
    return weight * similarity

def cosine_similarity_one_hot(cat_a, cat_b):
    return cosine_similarity(cat_a.reshape(1, -1), cat_b.reshape(1, -1))[0][0]

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

    # One-hot encode categorical features using the provided vm_categories dictionary
    enc = OneHotEncoder(sparse=False)
    for category, values in vm_categories.items():
        combined_df[category] = pd.Categorical(combined_df[category], categories=values)
    one_hot_encoded = enc.fit_transform(combined_df)
    one_hot_a = one_hot_encoded[0]
    one_hot_b = one_hot_encoded[1]

    # Calculate weighted similarity for each feature
    total_similarity = 0
    total_weight = sum(cat_feature_weights.values()) + sum(num_feature_weights.values()) + sum(hamming_feature_weights.values())

    # Categorical features
    one_hot_index = 0
    for feature, weight in cat_feature_weights.items():
        category_len = len(vm_categories[feature])
        feature_sim = cosine_similarity_one_hot(one_hot_a[one_hot_index:one_hot_index + category_len], one_hot_b[one_hot_index:one_hot_index + category_len])
        total_similarity += weight * feature_sim
        one_hot_index += category_len

    # Numerical features
    for feature, weight in num_feature_weights.items():
        feature_sim = weighted_similarity(float(df_a[feature]), float(df_b[feature]), weight, 'numerical')
        total_similarity += feature_sim

    # Hamming distance features
    for feature, weight in hamming_feature_weights.items():
        feature_sim = weighted_similarity(df_a[feature], df_b[feature], weight, 'hamming')
        total_similarity += feature_sim

    # Normalize the similarity score
    normalized_similarity = total_similarity / total_weight

    return normalized_similarity

# Predefined pool of categories and their possible values
vm_categories = {
    'cloud provider': ['Openstack', 'Cloudstack', 'AWS', 'GCP', 'Azure'],
    'OS_type': ['Ubuntu', 'CentOS', 'Debian', 'Windows'],
    'OS_version': ['20.04', '18.04', '10', '8'],
    'flavor_type': ['m1.small', 'm1.medium', 'm1.large', 'm1.xlarge'],
    'status': ['Running', 'Stopped', 'Paused', 'Suspended'],
    'region': ['RegionOne', 'RegionTwo']
}

cat_feature_weights = {
    'cloud provider': 0.2,
    'OS_type': 0.1,
    'OS_version': 0.1,
    'flavor_type': 0.15,
    'status': 0.1,
    'region': 0.2,
}

num_feature_weights = {
    'OS_version': 0.05,
    'flavor_vcpu': 0.1,
    'flavor_mem': 0.15,
    'volume_size': 0.15
}

hamming_feature_weights = {

}

# Example VM metadata
with open('metasim1.json') as m1:
    vm_metadata_a = json.load(m1)

with open('metasim2.json') as m2:
    vm_metadata_b = json.load(m2)

with open('metasim3.json') as m3:
    vm_metadata_c = json.load(m3)

similarity1 = metasim(vm_metadata_a, vm_metadata_b, cat_feature_weights, num_feature_weights, hamming_feature_weights, vm_categories)
similarity2 = metasim(vm_metadata_a, vm_metadata_c, cat_feature_weights, num_feature_weights, hamming_feature_weights, vm_categories)

print("Similarity a ~ b score:", similarity1)
print("Similarity a ~ c score:", similarity2)