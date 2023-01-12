from argparse import ArgumentParser
import re
import sys
from flask import request

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
    global parser
    parser = ParserPipeline.from_config(args.model_path)
    parser.load(args.model_path)


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

@app.get("api/parser")
def hello():
    return 200

@app.post("api/parser")
def apply_parser(r: ParserRequest):
    #docs = load_texts([r.text])
    #update_dataset_embeddings(docs, bert_model=args.bert_model)
    #doc = parser(docs[0])
    requestBody = request.get_json()
    r = requestBody['details']
    get_sentence_embeddings = get_sentence_embedder(args.bert_model)
    print(get_sentence_embeddings)
    sentence = ["translate pcm to english: " + str(r.text)]
    text_ = model.predict(sentence)
    # machine_translated = en2pcm.translate(text)
    machine_translated = text_[0]
    print(f"machine translated {machine_translated}")
    doc = add_parsers(tokenize(machine_translated))[0]
    # if len(doc.sentences) == 0:
    #     return
    # for sent_i, sent in enumerate(doc.sentences):
    #     sent_words = sent.tokens
    #     embeddings = get_sentence_embeddings(sent_words)
    #     doc.sentences[sent_i].embeddings = embeddings
    # doc = parser(doc)
    # print(doc.to_json())
    return({"translatedDetails": doc})
    # return doc.to_json()


@app.get("/api/parser/config")
def get_parser_config():
    configs = []
    for c in parser.components:
        configs.append(c.get_config())
    return configs


if __name__ == '__main__':
    uvicorn.run("app.run_bert_discourse_classifier_for_pcm:app", host=args.hostname, port=args.port, reload=args.reload, timeout_keep_alive=100 )
    # uvicorn.run(app, host=args.hostname, port=args.port, timeout_keep_alive=100 )
