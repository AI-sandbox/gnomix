# Temporary, just to get the point across and develop around how it will be

# Define a single source of gnomix data (obviously only works on galangal)
# this will be path to the hosted data
data_path="/home/wknd37/data"

# and here we would download the data and place it in gnomix/data
cp -r ${data_path} .
echo "Data downloaded and stored in gnomix/data/"