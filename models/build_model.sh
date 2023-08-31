# Copy the ipynb into the build directory
mkdir -p __build_model
cp models/pitchnet.ipynb __build_model/ || exit -2
cd __build_model || exit -1

# Set build flag
sed -i.back 's/build_network = False #/build_network = True #/g' pitchnet.ipynb
sed -i.back 's/activate_tensorboard = True #/activate_tensorboard = False #/g' pitchnet.ipynb
#sed -i.back 's/erase_present_datasets = False #/erase_present_datasets = True #/g' pitchnet.ipynb
#sed -i.back 's/reduce_dataset_size = True #/reduce_dataset_size = False #/g' pitchnet.ipynb
rm -f pitchnet.ipynb.back

# Run notebook in place
jupyter nbconvert --to notebook \
                  --execute pitchnet.ipynb \
                  --output pitchnet.ipynb \
                  --ExecutePreprocessor.timeout=36000

# Copy the ONNX export to GCP
ONNX_MODEL=`ls models/*.onnx | tail -n 1`
COMMIT_HASH=`git rev-parse --verify HEAD`
DATE_ID=$(date "+%Y%m%d")
DATA_BUILD_ID=`cat datasets_build_id`
ONNX_FILE_NAME=$DATE_ID"-"$COMMIT_HASH"-"$DATA_BUILD_ID".onnx"
gsutil cp $ONNX_MODEL "gs://models.pitchnet.app/builds/"$ONNX_FILE_NAME

# Convert ONNX to COREML and export it to GCP
COREML_FILE_NAME=$DATE_ID"-"$COMMIT_HASH"-"$DATA_BUILD_ID".mlmodel"
python3 ../models/onnx-to-coreml.py $ONNX_MODEL -o $COREML_FILE_NAME --minimum_ios_deployment_target 13
gsutil cp $COREML_FILE_NAME "gs://models.pitchnet.app/builds/"

# Zip build artifict and store them on GCP
rm -rf *.zip
rm -rf *Dataset
rm -f ../__build_model.zip
zip -r ../__build_model.zip .
cd ..


ZIP_FILE_NAME=$DATE_ID"-"$COMMIT_HASH"-"$DATA_BUILD_ID".zip"
gsutil cp __build_model.zip "gs://models.pitchnet.app/artifacts/"$ZIP_FILE_NAME

# Finaly remove the temporary directory
rm -rf __build_model
