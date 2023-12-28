import os
import numpy as np

import collections.abc
collections.MutableMapping = collections.abc.MutableMapping

from itertools import groupby
from operator import itemgetter
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, fields
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F
import torch


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

name_space = api.namespace('api', 'На вход поступает JSON, возвращает JSON')
input_data = name_space.model("Insert JSON",
                              {"texts": fields.List(fields.String(description="Insert texts", required=True)),
                               "score": fields.Float(description="Distance", required=True)}, )

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def cluster_name_number(vectors: np.array) -> np.array:
    """Function get vectors, finds vector most close to average of vectors and returns it's number."""
    # weight_average_vector = np.average(vectors, axis=0, weights=vectors)
    weight_average_vector = np.average(vectors, axis=0)
    weight_average_vector_ = weight_average_vector.reshape(1, weight_average_vector.shape[0])
    distances_from_average = cosine_similarity(vectors, weight_average_vector_)
    return np.argmax(distances_from_average)


def e5_vectorizer(texts: [str]):
    txts_chunks = chunks(texts, 5)
    vectors = []
    for num, txs in enumerate(txts_chunks):
        print(num + 1, "/", len(texts) // 5)
        batch_dict = e5_tokenizer(txs, max_length=512, padding=True, truncation=True, return_tensors='pt').to('cuda')
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        vectors += [torch.tensor(emb, device='cpu') for emb in  embeddings]
    return vectors


def grouped_func(data: list) -> [{}]:
    """Function groups input list of data with format: [(label, vector, text)]
    into list of dictionaries, each dictionary of type:
    {
    label: label,
    texts: list of texts correspond to label
    vectors_matrix: numpy matrix of vectors correspond to label
    }
    """
    data = sorted(data, key=lambda x: x[0])
    grouped_data = []
    for key, group_items in groupby(data, key=itemgetter(0)):
        d = {"label": key, "texts": []}
        temp_vectors = []
        for item in group_items:
            temp_vectors.append(item[1])
            d["texts"].append(item[2])
        d["vectors_matrix"] = np.vstack(temp_vectors)
        grouped_data.append(d)
    return grouped_data


def clustering_func(vectorizer, clusterer: AgglomerativeClustering, texts: []) -> {}:
    """Function for text collection clustering"""
    vectors = vectorizer(texts)
    # print(vectors)
    
    # np_array = np.array([t_v.numpy().reshape(1024) for t_v in vectors])
    np_array = np.array(vectors)
    print(np_array.shape)
    clusters = clusterer.fit(vectors)
    data = [(lb, v, tx) for lb, v, tx in zip(clusters.labels_, vectors, texts)]
    grouped_data = grouped_func(data)
    result_list = []
    for d in grouped_data:
        label = str(d["label"])
        title_number = cluster_name_number(d["vectors_matrix"])
        title = d["texts"][title_number]
        cluster_size = len(d["texts"])
        result_list += [(label, title, tx, cluster_size) for tx in d["texts"]]
    return {"texts_with_labels": result_list}

e5_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to('cuda')

@name_space.route('/clusterer')
class Clustering(Resource):
    @name_space.expect(input_data)
    def post(self):
        """POST method on input csv file with texts and score, output clustering texts as JSON file."""
        json_data = request.json
        texts_list = json_data["texts"]

        """restricting number of texts fragments (resource limit)"""
        clustering_texts = texts_list[:30000]
        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=json_data['score'],
                                            memory=os.path.join("cache"))


        return jsonify(clustering_func(e5_vectorizer, clusterer, clustering_texts))


if __name__ == "__main__":
    # serve(app, host="0.0.0.0", port=8080)
    app.run(host='0.0.0.0', port=4500)
