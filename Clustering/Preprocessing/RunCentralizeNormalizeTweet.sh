data_types=(
    "TweetTopic"
    "TweetFinance" 
    "AgNews"
    "AgNewsTitle"
    "20News"
    )

vectorize_type=sentenceBERT

transformer_models=(
    sentence-transformers/all-MiniLM-L6-v2
    sentence-transformers/all-mpnet-base-v2
    sentence-transformers/all-distilroberta-v1
)


for data_type in ${data_types[@]}; do
    for transformer_model in ${transformer_models[@]}; do
        echo "start ${data_type} ${transformer_model}"
        python CentralizeNormalize.py "${data_type}" "${vectorize_type}" "${transformer_model}"
    done
done

