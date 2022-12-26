mkdir -p dataset/cats_dogs
wget -c https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip -O dataset/cats_dogs/kagglecatsanddogs_5340.zip
unzip -o dataset/cats_dogs/kagglecatsanddogs_5340.zip -d dataset/cats_dogs
find dataset/cats_dogs/PetImages/ -type f -not -name "*.jpg" -delete
