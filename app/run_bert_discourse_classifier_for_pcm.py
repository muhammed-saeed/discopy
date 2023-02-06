from argparse import ArgumentParser
import re
import sys
from flask import request
import json
import trankit

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic.main import BaseModel

from discopy.parsers.pipeline import ParserPipeline
#from discopy_data.data.loaders.raw import load_texts
#from discopy_data.data.update import update_dataset_embeddings
from discopy_data.nn.bert import get_sentence_embedder


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import logging
# import sacrebleu
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args = T5Args()
model_args.max_length = 256
model_args.n_gpu = 4
model_args.length_penalty = 1
model_args.num_beams = 10
model_args.use_multiprocessed_decoding = False
# model_output_dir = "/home/mohammed_yahia3/checkpoint-49960-epoch-8"
model_output_dir="/home/CE/musaeed/checkpoint-185000-epoch-9"
model = T5Model("t5", model_output_dir, args=model_args, use_cuda=True)



arg_parser = ArgumentParser()
arg_parser.add_argument("--hostname", default="127.0.0.1", type=str, help="REST API hostname")
arg_parser.add_argument("--port", default=8081, type=int, help="REST API port")
arg_parser.add_argument("--model-path", type=str, help="path to trained discourse parser")
arg_parser.add_argument("--bert-model", default='bert-base-cased', type=str, help="bert model name")
arg_parser.add_argument("--reload", action="store_true", help="Reload service on file changes")
args = arg_parser.parse_args()

model_path = "/home/CE/musaeed/bert_model/"

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser: ParserPipeline = None
get_sentence_embeddings = None

@app.on_event("startup")
async def startup_event():
    print("you are welcome here")
    global parser
    global parserForLoadText
    parser = ParserPipeline.from_config(args.model_path)
    parser.load(args.model_path)
    # parser = ParserPipeline.from_config(model_path)
    # parser.load(model_path)
    tmp_stdout = sys.stdout
    sys.stdout = sys.stderr
    parserForLoadText = trankit.Pipeline('english', cache_dir=os.path.expanduser('~/.trankit/'), gpu=False)
    parserForLoadText.tokenize("Init")
    sys.stdout = tmp_stdout
    print("we have completed the load_paraser step", file=sys.stderr)
    # parserForLoadText


class ParserRequest(BaseModel):
    details: str
    title: str

# def load_parser(use_gpu=False):
#     import trankit
#     tmp_stdout = sys.stdout
#     sys.stdout = sys.stderr
#     parser = trankit.Pipeline('english', cache_dir=os.path.expanduser('~/.trankit/'), gpu=use_gpu)
#     parser.tokenize("Init")
#     sys.stdout = tmp_stdout
#     print("we have completed the load_paraser step", file=sys.stderr)
#     return parser

# parserForLoadText = load_paraser()

def tokenize(text, fast = True, tokenize_only = False):
    # from discopy_data.data.loaders.raw import load_texts, load_texts_fast
    from discopy_data.data.loaders.raw import load_textss as load_texts_fast
    #just make a copy of load_texts and call it load_textss
    from discopy_data.data.loaders.raw import load_texts
    output = []
    arr2text = ". ".join(text)
    print(f"the input text to the tokenize is {text}",file = sys.stderr)
    
    document_loader = load_texts_fast # if fast else load_texts
    for doc in document_loader(re.split(r'\n\n\n+', text), parserForLoadText, tokenize_only=tokenize_only):
        output.append(doc)
    print(f'the tokenized text is {output[0].to_json()} ', file=sys.stderr)
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
    print(f"the input to the add_parses is {src[0].to_json()}", file = sys.stderr)
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
    print(f"the output to the add_parses is {output[0].to_json()}", file = sys.stderr)
    return output


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/api/parser")
def hello():
    return 200

@app.post("/api/parser")
def apply_parser(r: ParserRequest):
    
    get_sentence_embeddings = get_sentence_embedder(args.bert_model)
    # get_sentence_embeddings = get_sentence_embedder(model_path)
    print(f"{get_sentence_embeddings}",file=sys.stderr)
    # print(f"Input text {r.details}")
    pcmTEXT = r.details
    text_array = pcmTEXT.split(".")
    text_array = list(filter(lambda x: x.strip(), text_array))

    sentences = [ "translate pcm to english: " +r for r in text_array]
    print(f"{sentences}", file=sys.stderr)
    text_ = model.predict(sentences)
    # machine_translated = en2pcm.translate(text)
    machine_translated = text_
    translation = ". ".join(machine_translated) + "."
    print(f"input text is {pcmTEXT}", file=sys.stderr)
    print(f"machine translated {translation}", file=sys.stderr)
    doc = add_parsers(tokenize(str(translation)))[0]
    if len(doc.sentences) == 0:
        return({"translatedDetails": str("You are passing empty string ;)")})

    for sent_i, sent in enumerate(doc.sentences):
        sent_words = sent.tokens
        embeddings = get_sentence_embeddings(sent_words)
        doc.sentences[sent_i].embeddings = embeddings
    doc = parser(doc)
    print(doc.to_json())
    # doc = json.dumps(doc)
    doc_json = doc.to_json()
    # return({"translatedDetails": doc})
    return({"translatedDetails": str(doc_json)})
    # return doc.to_json()


@app.post("/api/parseren")
def apply_parser(r: ParserRequest):
    # docs = load_texts([r.details])
    # update_dataset_embeddings(docs, bert_model=args.bert_model)
    # update_dataset_embeddings(docs, bert_model=model_path)
    # doc = parser(docs[0])
    # return 200;
    # r = requestBody['details']
    get_sentence_embeddings = get_sentence_embedder(args.bert_model)
    # get_sentence_embeddings = get_sentence_embedder(model_path)
    print(f"get_sentence_embeddings", file = sys.stderr)
    # sentence = ["translate pcm to english: " + str(r.details)]
    # text_ = model.predict(sentence)
    # machine_translated = en2pcm.translate(text)
    # machine_translated = text_[0]
    print(f"WE are using the english parser and we are here", file=sys.stderr)
    enTEXT = r.details
    text_array = enTEXT.split(".")
    text_array = list(filter(lambda x: x.strip(), text_array))
    translation = ". ".join(text_array) + "."
    print(f"The english text to be processes is {translation}")
    doc = add_parsers(tokenize(str(translation)))[0]
    if len(doc.sentences) == 0:
        return({"translatedDetails": str("You are passing empty string ;)")})
    for sent_i, sent in enumerate(doc.sentences):
        sent_words = sent.tokens
        embeddings = get_sentence_embeddings(sent_words)
        doc.sentences[sent_i].embeddings = embeddings
    doc = parser(doc)
    print(doc.to_json(), file=sys.stderr)
    # doc = json.dumps(doc)
    doc_json = doc.to_json()
    # return({"translatedDetails": doc})
    return({"translatedDetails": str(doc_json)})
    # return doc.to_json()


@app.get("/api/parser/config")
def get_parser_config():
    configs = []
    for c in parser.components:
        configs.append(c.get_config())
    return configs


if __name__ == '__main__':
    uvicorn.run("run_bert_discoures_classifier:app", host=args.hostname, port=args.port, reload=args.reload, timeout_keep_alive=100 )
    # uvicorn.run("run_bert_discoures_classifier:app",  reload=args.reload, timeout_keep_alive=100 )

    # uvicorn.run("app", host=args.hostname, port=args.port, reload=args.reload, timeout_keep_alive=100 )
    # uvicorn.run(app, host=args.hostname, port=args.port, timeout_keep_alive=100 )
