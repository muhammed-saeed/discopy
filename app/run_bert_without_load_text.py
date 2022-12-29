from argparse import ArgumentParser
import re
import sys

import uvicorn
from fastapi import FastAPI
from pydantic.main import BaseModel

from discopy.parsers.pipeline import ParserPipeline
#from discopy_data.data.loaders.raw import load_texts
#from discopy_data.data.update import update_dataset_embeddings
from discopy_data.nn.bert import get_sentence_embedder


arg_parser = ArgumentParser()
arg_parser.add_argument("--hostname", default="0.0.0.0", type=str, help="REST API hostname")
arg_parser.add_argument("--port", default=8081, type=int, help="REST API port")
arg_parser.add_argument("--model-path", type=str, help="path to trained discourse parser")
arg_parser.add_argument("--bert-model", default='bert-base-uncased', type=str, help="bert model name")
arg_parser.add_argument("--reload", action="store_true", help="Reload service on file changes")
args = arg_parser.parse_args()

app = FastAPI()
parser: ParserPipeline = None
get_sentence_embeddings = None

@app.on_event("startup")
async def startup_event():
    global parser
    parser = ParserPipeline.from_config(args.model_path)
    parser.load(args.model_path)


@app.get("/api/parser/config")
def get_parser_config():
    configs = []
    for c in parser.components:
        configs.append(c.get_config())
    return configs


class ParserRequest(BaseModel):
    text: str


def tokenize(text, fast = True, tokenize_only = True):
    from discopy_data.data.loaders.raw import load_texts, load_texts_fast
    output = []
    document_loader = load_texts_fast if fast else load_texts
    for doc in document_loader(re.split(r'\n\n\n+', text), tokenize_only=tokenize_only):
        output.append(doc)
    return output

def add_parsers(src, 
    constituent_parser='crf-con-en', 
    dependency_parser='biaffine-dep-en', 
    constituents=True, 
    dependencies=True, 
):
    import supar
    from discopy_data.data.update import get_constituent_parse, get_dependency_parse
    sys.stderr.write('SUPAR load constiuent parser!\n')
    cparser = supar.Parser.load(constituent_parser) if constituents else None
    sys.stderr.write('SUPAR load dependency parser!\n')
    dparser = supar.Parser.load(dependency_parser) if dependencies else None
    output = []
    for doc in src:
        for sent_i, sent in enumerate(doc.sentences):
            inputs = [(t.surface, t.upos) for t in sent.tokens]
            if cparser:
                parsetree = get_constituent_parse(cparser, inputs)
                doc.sentences[sent_i].parsetree = parsetree
            if dparser:
                dependencies = get_dependency_parse(dparser, inputs, sent.tokens)
                doc.sentences[sent_i].dependencies = dependencies
        output.append(doc)
    sys.stderr.write('Supar parsing done!\n')
    return output



@app.post("/api/parser")
def apply_parser(r: ParserRequest):
    #docs = load_texts([r.text])
    #update_dataset_embeddings(docs, bert_model=args.bert_model)
    #doc = parser(docs[0])
    get_sentence_embeddings = get_sentence_embedder(args.bert_model)
    print(get_sentence_embeddings)
    doc = add_parsers(tokenize(r.text))[0]
    if len(doc.sentences) == 0:
        return
    for sent_i, sent in enumerate(doc.sentences):
        sent_words = sent.tokens
        embeddings = get_sentence_embeddings(sent_words)
        doc.sentences[sent_i].embeddings = embeddings
    doc = parser(doc)
    print(doc.to_json())
    return doc.to_json()


if __name__ == '__main__':
    uvicorn.run("app.run_bert:app", host=args.hostname, port=args.port, reload=args.reload)
