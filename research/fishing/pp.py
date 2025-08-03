import spacy

text_fr = "La bataille d'El-Alamein en Égypte oppose la 8e armée britannique dirigée par Bernard Montgomery aux divisions d'Erwin Rommel."
nlp_model_fr = spacy.load("fr_core_news_sm")
nlp_model_fr.add_pipe("entityfishing", config={"language": "fr"})
doc_fr = nlp_model_fr(text_fr)

options = {
    "ents": ["MISC", "LOC", "PER"],
    "colors": {"LOC": "#82e0aa", "PER": "#85c1e9", "MISC": "#f0b27a"}
}

params = {"text": doc_fr.text,
          "ents": [{"start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_,
                    "kb_id": ent._.kb_qid,
                    "kb_url": ent._.url_wikidata}
                     for ent in doc_fr.ents],
          "title": None}

spacy.displacy.serve(params, style="ent", manual=True, options=options)
