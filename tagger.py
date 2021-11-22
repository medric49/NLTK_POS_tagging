from nltk.tag import TaggerI
import spacy.tokens


class SpacyTagger(TaggerI):

    def __init__(self):
        super(SpacyTagger, self).__init__()

        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def tag(self, tokens):
        doc = spacy.tokens.doc.Doc(self.nlp.vocab, words=tokens)

        for _, proc in self.nlp.pipeline:
            doc = proc(doc)

        return [(t.text, t.tag_) for t in doc]
