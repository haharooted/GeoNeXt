# GeoneXt

### Goal:
- Given a news article or tweet, figure out locations in this tweet. Most approaches give one location, i think we want multiple

### Broad categories:
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
    US focused
- GeoWebNews
    US focused, broader EU coverage
- TR-News
    EU, ME, East Asia, Australia
- 

### Other approaches:
- Edinburgh (Grover et al. 2010)
    Rule-based extraction and disambiguation system
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


- ReFinED (Ayoola et al., 2022a)
    Introduced a vector-space approach for joint extraction and disambiguation of Wikipedia entities. One transformer network generates contextualized embeddings for tokens in the text, another generates embeddings for entries in the ontology, and tokens are matched to entries by comparing dot products over embeddings. ReFinED was trained on Wikipedia, and Wikipedia entries for place names have GeoNames IDs, so ReFinED can be used as a geocoder.


- GeoNorm (Zhang, et al., 2023)
    a BERT-based transformer model is employed to rerank location candidates, using contextual embeddings to prioritise candidates that best match a toponym’s context.


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




