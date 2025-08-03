# GeoneXt
![Showcase](/pictures/Excalidraw_GEONEXT_3.png)

### Goal:
- Given a news article or tweet, figure out locations in this tweet. Most approaches give one location, i think we want multiple. Toponym resolution is crucial for extracting geographic information from natural language texts, such as social media posts and news articles.

### Broad categories:
Geoparsing consists of two steps: toponym recognition, which is to recognize toponyms mentioned in texts, and toponym resolution or geocoding, which is to determine the geospatial representation or geo-coordinates of the toponyms.
- Toponym recognition / Geoparsing
- Toponym resolution / Geolocating/geocoding

A geocoder is a large database with many addresses and place named indexed for search, examples:
- Nominatim
- Geonames
- Photon
- Pelias
- (Proprietary) Google
- (Proprietary) Bing
- (Proprietary) Foursquare




### Types of geocoding:
- Mention-level geocoding (Single post)
- Document-level geocoding (Twitter users, news)

### Datasets
#### News Articles:
- Local Global Lexicon
    LGL19 (Local-Global Lexicon) corpus was created by Lieberman et al. (2010), containing 588 human-annotated news articles published by 78 local newspapers.
- GeoWebNews
    US focused, broader EU coverage
    was created by Gritta et al. (2018a) from news articles collected from April 1st to 8th in 2018.
- TR-News
    EU, ME, East Asia, Australia
    TR-News was created by Kamalloo and Rafiei (2018) from news articles of different sources.
- TopRes19th
    455 News articles in which places were manually annotated and linked to Wikipedia (mapped to Wikidata)
- Hipe2020
    News articles in English, French, German
    Linked whenever possible to their corresponding Wikidata (and therefore locations)
- AIDA
- NEEL
- WOTR
- WikToR
    WikToR was created by Gritta et al. (2018b) in an automatic manner, containing 5,000 Wikipedia articles with many ambiguous places, such as (Santa Maria, California), (Santa Maria, Bulacan), (Santa Maria, Ilocos Sur), and (Santa Maria, Romblon).
- GeoVirus
    GeoVirus was created by Gritta et al. (2018a), containing news articles about epidemics, such as Ebola and Swine Flu.
- GeoCorpora
    GeoCorpora was created by Wallgrün et al. (2018) containing tweets related to multiple events (e.g., ebola, flood, and rebel) that happened across the world in 2014 and 2015.
    
- CLDW
- SemEval-2019
- NCEN


### Other approaches:
- Edinburgh (Grover et al. 2010)
    Rule-based extraction and disambiguation system developed by the Language Technology Group (LTG) at Edinburgh University. Provided code and API.
    https://royalsocietypublishing.org/doi/10.1098/rsta.2010.0149

- Mordecai (Halterman, 2017)
    Generate-and-rank approach that uses Elasticsearch to generate candidates and neural networks based on word2vec (Mikolov et al., 2013) -  trained on proprietary data
    https://joss.theoj.org/papers/10.21105/joss.00091

- CamCoder (Gritta et al., 2018)
    Tile-classification approach that combines CNN over the target mention and 400 tokens of context with a population vector from location mentions in GeoNames, it then predicts one of 7823 tiles on the earth.
    https://aclanthology.org/P18-1119.pdf

- DeezyMatch (Liu et al., 2021)
    Vector-space approach that first pretrains a LSTM-based classifier on GeoNames taking string pairs as input and then fine-tunes the pair classifier on the target dataaset. It then compares mentions to database entries by generating vector representations for both and measuring their cosine similarity.

- SAPBERT (Liu et al., 2021)
    Vector-space approach that pretrains a transformer network on the database using a self-alignment metric learning objective and online hard pairs mining to cluster synonyms of the same concept together and move different concepts further away, then its finetuned on the target dataset. Originally used for bio domain, but can be retrained for other domains.
    https://arxiv.org/pdf/2010.11784

- BLINK (Wu et al., 2020b)
    Entity Linking

- GENRE (De Cao et al., 2021)
    Entity lnker

- ReFinED (Ayoola et al., 2022a)
    Introduced a vector-space approach for joint extraction and disambiguation of Wikipedia entities. One transformer network generates contextualized embeddings for tokens in the text, another generates embeddings for entries in the ontology, and tokens are matched to entries by comparing dot products over embeddings. ReFinED was trained on Wikipedia, and Wikipedia entries for place names have GeoNames IDs, so ReFinED can be used as a geocoder.

- GeoNorm (Zhang, et al., 2023)
    a BERT-based transformer model is employed to rerank location candidates, using contextual embeddings to prioritise candidates that best match a toponym’s context.

- T-Res (Ardanuy, et. al., 2023)
    BERT based approach

- LUKE
    Entity disambiguation model based on BERT

- Adaptive Learning
    Adaptive Learning is a random forest-based toponym resolution approach. 

- Nominatim geocoder
    Geocoder based on OpenStreetMap


- GeoLM 
    BERT based, looks like
    

- Voting approach
    Multiple people have tried combining multiple existing toponymn recognition approaches, and getting good performance with using DBSCAN for voting.



### Suggestions for evaluation metrics
- Accuracy
Accuracy is the number of location mentions
where the system predicted the correct database entry ID, divided by the number of location mentions.
Higher is better, and a perfect model would have
accuracy of 1.0.

- Accuracy@161km 
Accuracy@161km measures the fraction of
system-predicted (latitude, longitude) points that
were less than 161 km (100 miles) away from
the human-annotated (latitude, longitude) points.
Higher is better, and a perfect model would have
Accuracy@161km of 1.0.

- Mean Error distance
Mean error distance calculates the mean over
all predictions of the distance between each systempredicted and human-annotated (latitude, longitude) point. Lower is better, and a perfect model
would have a mean error distance of 0.0.

- Area Under the Curve
Area Under the Curve calculates the area under
the curve of the distribution of geocoding error
distances. Lower is better, and a perfect model
would have an area under the curve of 0.0.




