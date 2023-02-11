data_types=(
    "20News" 
    )

sampling_nums=(
    1 2 4 8 16 32 64 128
)


for data_type in ${data_types[@]}; do
    for sampling_num in ${sampling_nums[@]}; do
        echo "start ${data_type}${sampling_num}"
        python LDA.py "${data_type}Sampled${sampling_num}" 
    done
done


