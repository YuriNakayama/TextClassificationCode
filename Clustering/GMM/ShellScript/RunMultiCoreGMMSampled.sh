data_types=(
    "20News" 
    )

sampling_nums=(
    1 2 4 8 16 32 64 128
)

transformer_models=(
    sentence-transformers/all-MiniLM-L6-v2
    sentence-transformers/all-mpnet-base-v2
    sentence-transformers/all-distilroberta-v1
)


for data_type in ${data_types[@]}; do
    for transformer_model in ${transformer_models[@]}; do
        for sampling_num in ${sampling_nums[@]}; do
            echo "start ${data_type}${sampling_num} ${transformer_model}"
            python ../MultiCoreGMM.py "${data_type}Sampled${sampling_num}" "sentenceBERT" "${transformer_model}"
        done
    done
done

