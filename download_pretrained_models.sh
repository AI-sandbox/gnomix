## code adapted from https://stackoverflow.com/questions/48133080/how-to-download-a-google-drive-url-via-curl-or-wget
fileid="1Q0zg9zqTaZUvt42uxzE_0gcfnFvjwi33"
filename="pretrained_gnomix_models.tar.gz" # outputs in the same folder
echo "Download in progress..."
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
rm cookie
# untar the root dir
echo "Expanding files..."
tar -xvf pretrained_gnomix_models.tar.gz
rm pretrained_gnomix_models.tar.gz
echo "Downloaded pretrained models to directory ./pretrained_gnomix_models"