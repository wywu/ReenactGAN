transformer=$1
decoder=$2
who=$3

cp $transformer ./pretrained_models/$who"_net_transformer.pth"
cp $decoder ./pretrained_models/$who"_net_decoder.pth"
