TROJAI_R6_DATASET_DIR="/data/share/trojai/trojai-round6-v2-dataset/"
for n in {12..17} {36..41}
do
    echo $n && python nlp_ami.py --model_filepath=${TROJAI_R6_DATASET_DIR}/models/id-$(printf %08d $n)/model.pt --examples_dirpath=${TROJAI_R6_DATASET_DIR}/models/id-$(printf %08d $n)/poisoned_example_data #| grep "total correct predictions"
done


#-----------------------
#BERT
#-----------------------

#for n in {42..47} {18..23}
#do
#    echo $n && python nlp_ami.py  --tokenizer_filepath=${TROJAI_R6_DATASET_DIR}/tokenizers/DistilBERT-distilbert-base-uncased.pt --model_filepath=${TROJAI_R6_DATASET_DIR}/models/id-$(printf %08d $n)/model.pt --examples_dirpath=${TROJAI_R6_DATASET_DIR}/models/id-$(printf %08d $n)/clean_example_data | grep "total correct predictions"
#done

#echo "_________"

#for n in {6..11} {30..35}
#do
#    echo $n && python nlp_ami.py  --tokenizer_filepath=${TROJAI_R6_DATASET_DIR}/tokenizers/DistilBERT-distilbert-base-uncased.pt --model_filepath=${TROJAI_R6_DATASET_DIR}/models/id-$(printf %08d $n)/model.pt --examples_dirpath=${TROJAI_R6_DATASET_DIR}/models/id-$(printf %08d $n)/clean_example_data | grep "total correct predictions"
#done
