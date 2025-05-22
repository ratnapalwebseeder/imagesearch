# Image-search-engine

we are currently using `resenet18` pretrained model of CNN algorithm.

once we got labelled data then we can derive the accuracy of our model and will also compare more advance models like `resenet50` and `EfficientNetB2`.

if still accuracy is not good then we can hypertune the model using our own data.

### usage :

first we need to install required packages so run this command
```bash
pip install -r requirements.txt
```

**step 1 :**
```bash
python main.py
````

This command will create vector database for the images in the folder `software_data`.

**step 2 :**

```bash
streamlit run --server.fileWatcherType=none frontend.py
```

Streamlit provides basic frontend to demonstrate the working of program.

open browser and go to `localhost:8501` upload any one refference image from the images folder or any image you want

Output will be the most top 5 similar images from the dataset of our metal designs.
below the every output image you will see the similarity score their.
