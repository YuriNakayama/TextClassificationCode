config = {
    "data": {
        "AgNews": {"class_num": [2, 4, 8, 16, 32, 64]},
        "AgNewsTitle": {
            "class_num": [
                2,
                4,
                8,
                16,
                32,
                64,
            ]
        },
        "20News": {"class_num": [5, 10, 20, 40, 80]},
    },
    "vectorize": {
        "doc2vec": {
            "max_model_num": 1,
            "normalization": ["normalized", "centralized"],
            "dims": [2, 4, 8, 16, 32, 64, 128, 256],
        },
        "sentenceBERT": {
            "sentence-transformers/all-MiniLM-L6-v2": {
                "max_model_num": 1,
                "normalization": ["normalized", "centralized"],
                "dims": [2, 4, 8, 16, 32, 64, 128, 256, 384],
            },
            "sentence-transformers/all-mpnet-base-v2": {
                "max_model_num": 1,
                "normalization": ["normalized", "centralized"],
                "dims": [2, 4, 8, 16, 32, 64, 128, 256, 768],
            },
            "sentence-transformers/all-distilroberta-v1": {
                "max_model_num": 1,
                "normalization": ["normalized", "centralized"],
                "dims": [2, 4, 8, 16, 32, 64, 128, 256, 768],
            },
        },
    },
    "clustering": {
        "gmm": {
            "max_model_num": 30,
            "covariance_types": ["spherical", "diag", "full"],
        },
        "kmeans": {
            "max_model_num": 30,
        },
        "LDA": {
            "max_model_num": 30,
        },
    },
}

# +
{
    
}
# -

data_type_re = {
    "20News": r"^20News",
    "AgNewsTitle": r"^AgNewsTitle",
    "AgNews": r"^AgNews",
}

# +
# config = {
#     "data": {
#         "AgNews": {
#             "class_num": 4,
#         },
#         "AgNewsTitle": {
#             "class_num": 4,
#         },
#         "20News": {"class_num": 20},
#     },
#     "vectorize": {
#         "doc2vec": {
#             "max_model_num": 1,
#             "normalization": "normalized",
#             "dims": [2, 4, 6, 8, 10, 20, 40, 80, 160],
#         },
#         "sentenceBERT": {
#             "max_model_num": 1,
#             "normalization": "normalized",
#             "dims": [2, 4, 6, 8, 10, 20, 40, 80, 160, 384],
#         },
#     },
#     "clustering": {
#         "gmm": {
#             "max_model_num": 30,
#             "covariance_types": ["spherical", "diag", "tied", "full"],
#         },
#         "kmeans": {
#             "max_model_num": 30,
#         },
#         "LDA": {
#             "max_model_num": 30,
#         },
#     },
# }
# -


