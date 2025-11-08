
#!/bin/bash

OUTPUT_PATH="src/advcloak/"
OUTPUT_NAME="ser50_webface_soft"
PATH_TO_SYMBOL="model_checkpoints/models/target/$OUTPUT_NAME/model-symbol.json"
PATH_TO_PARAMS="model_checkpoints/models/target/$OUTPUT_NAME/model-0025.params"

echo "---- Obtaining IRs -----"
python -m mmdnn.conversion._script.convertToIR -f mxnet -d $OUTPUT_PATH$OUTPUT_NAME -w $PATH_TO_PARAMS -n $PATH_TO_SYMBOL --inputShape 3,112,112

echo "---- Converting IRs -----"
python -m mmdnn.conversion._script.IRToCode -f pytorch -n $OUTPUT_PATH$OUTPUT_NAME.pb -w $OUTPUT_PATH$OUTPUT_NAME.npy -d $OUTPUT_PATH$OUTPUT_NAME.py -dw $OUTPUT_PATHmodel_$OUTPUT_NAME.pth
