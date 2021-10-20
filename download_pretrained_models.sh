## code adapted from https://stackoverflow.com/questions/48133080/how-to-download-a-google-drive-url-via-curl-or-wget
fileid="14D_KiJSgW-Q36B-8jky7JdQUTvUj_tXC"
filename="pretrained_gnomix_models.tar.gz" # outputs in the same folder
echo "Download in progress..."
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm cookie
# untar the root dir
echo "Expanding files..."
tar -xvf pretrained_gnomix_models.tar.gz
rm pretrained_gnomix_models.tar.gz
echo "Downloaded pretrained models to directory ./pretrained_gnomix_models"