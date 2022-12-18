config = {
    "data": {
        "AgNews": {
            "class_num": 4,
        },
        "AgNewsTitle": {
            "class_num": 4,
        },
        "20News": {"class_num": 20},
    },
    "vectorize": {
        "doc2vec": {
            "max_model_num": 1,
            "dims": [2, 4, 6, 8, 10, 20, 40, 80, 160],
        },
        "sentenceBERT": {
            "max_model_num": 1,
            "dims": [2, 4, 6, 8, 10, 20, 40, 80, 160],
        },
    },
    "clustering": {
        "gmm": {
            "max_model_num": 30,
            "covariance_types": ["spherical", "diag", "tied", "full"],
        },
        "kmeans": {
            "max_model_num": 30,
        },
        "LDA": {
            "max_model_num": 30,
        },
    },
}


