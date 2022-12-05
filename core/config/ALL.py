config = {
    "vectorize": {
        "doc2vec": {
            "max_model_num": 0,
            "dims": [2, 3, 4, 6, 7, 8, 9, 10, 20, 40, 80, 160, 320, 640],
        }
    },
    "clustering": {
        "gmm": {
            "max_model_num": 2,
            "covariance_types": ["spherical", "diag", "tied", "full"],
        },
        "kmeans": {
            "max_model_num": 2,
        },
    },
}


