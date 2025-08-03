import spacy
from spacy.tokens import Span

def __call__(self, doc):
        results = self.main_disambiguation_process_batch(doc)

        if not results:
            print("[WARNING] No results from entity-fishing disambiguation.")
            return doc

        result = results[0]  # safe now

        for ent_data in result.get("entities", []):
            start = ent_data.get("start_char")
            end = ent_data.get("end_char")
            kb_qid = ent_data.get("wikidata_id")
            url = ent_data.get("wikidata_url")
            score = ent_data.get("score")

            if start is None or end is None:
                continue

            span = doc.char_span(start, end, alignment_mode="contract")
            if span:
                span._.kb_qid = kb_qid
                span._.url_wikidata = url
                span._.nerd_score = score

        return doc


# Register extensions (in case not already registered in the pipeline)
if not Span.has_extension("kb_qid"):
    Span.set_extension("kb_qid", default=None)
if not Span.has_extension("url_wikidata"):
    Span.set_extension("url_wikidata", default=None)
if not Span.has_extension("nerd_score"):
    Span.set_extension("nerd_score", default=None)

# Input text
text_en = "Victor Hugo and Honoré de Balzac are French writers who lived in Paris."

# Load spaCy model
nlp_model_en = spacy.load("en_core_web_sm")

# Preview detected entities before adding entityfishing
print("Entities before entityfishing:")
doc_before = nlp_model_en(text_en)
for ent in doc_before.ents:
    print(ent.text, ent.label_)

# Add entityfishing pipeline component
try:
    nlp_model_en.add_pipe("entityfishing")
except Exception as e:
    print(f"[ERROR] Failed to add 'entityfishing' pipe: {e}")
    exit(1)


doc_en = nlp_model_en(text_en)
print("\nEntities after entityfishing:")
for ent in doc_en.ents:
        print((
                ent.text,
                ent.label_,
                ent._.kb_qid,
                ent._.url_wikidata,
                ent._.nerd_score
        ))
