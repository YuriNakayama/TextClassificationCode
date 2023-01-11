data_types=(
    "AgNews" 
    "20News" 
    "AgNewsTitle"
    )
vectorize_types=("sentenceBERT")

for data_type in ${data_types[@]}; do
    for vectorize_type in ${vectorize_types}; do
        echo "start ${data_type} ${vectorize_type}"
        python GMM.py ${data_type} ${vectorize_type}
    done
done
