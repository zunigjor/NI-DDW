import json
import nltk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import wikipedia
import sys
import warnings



########################################################################################################################
# TEXT MINING
########################################################################################################################
def pos_tagging(text):
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sent) for sent in sentences]
    tagged = [nltk.pos_tag(sent) for sent in tokens]
    return tagged


def ner_entity_classification(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    ne_chunked = nltk.ne_chunk(tagged)
    data = extract_entities(ne_chunked)
    return data


def extract_entities(ne_chunked):
    # Extract entities
    data = {}
    for entity in ne_chunked:
        if isinstance(entity, nltk.tree.Tree):
            text = " ".join([word for word, tag in entity.leaves()])
            ent = entity.label()
            if text not in data:
                data[text] = [ent, 0]
            data[text][1] += 1
        else:
            continue
    # Transofrm data into a list of tuples (triples)
    list_of_tuples = [(k, v[0], v[1]) for k, v in data.items()]
    # Sort
    sorted_data = sorted(list_of_tuples, key=lambda x: x[2], reverse=True)
    return sorted_data


def ner_custom_pattern(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    data = {}
    entity = []
    for tagged_entry in tagged:
        # all adjectives and optionally plus noun
        if tagged_entry[1].startswith("JJ") or (entity and tagged_entry[1].startswith("NN")):
            entity.append(tagged_entry)
        else:
            if entity:
                # get rid of not real adjectives
                if len(entity[0][0]) <= 2:
                    entity = []
                else:
                    word = " ".join(e[0] for e in entity)
                    if word not in data:
                        data[word] = [entity[0][1], 0]
                    data[word][1] += 1
                    entity = []
    # Transform data into list of triples
    list_of_tuples = [(k, v[0], v[1]) for k, v in data.items()]
    # Sort
    sorted_data = sorted(list_of_tuples, key=lambda x: x[2], reverse=True)
    return sorted_data


def ner_hugging_face(text):
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(text)
    results = []
    for entity in ner_results:
        results.append((entity['word'], entity['entity']))
    return results


def wiki_definition(entity):
    # Get first sentence from wikipedia summary and tag it, catch errors
    try:
        summary = wikipedia.summary(entity, sentences=1, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        # This error happens when wiki has more options for a word, choose the first option
        try:
            summary = wikipedia.summary(e.options[0], sentences=1, auto_suggest=False)
        except Exception:
            # This can still fail. So categorize as Thing
            return "Thing"
    except Exception:
        # Errors happen when wikipedia has no result. So just categorize it as a Thing
        return "Thing"
    summary_sentence = nltk.pos_tag(nltk.word_tokenize(summary))
    # Detect pattern of describing what given entity is
    grammar = "NP: {<DT>?<JJ>?<VBN>*<NN>?<NN>}"
    cp = nltk.RegexpParser(grammar)
    chunked = cp.parse(summary_sentence)
    for word in chunked:
        # Skip first (Repeats entity name)
        if word == chunked[0] or len(word) < 2:
            continue
        if isinstance(word, nltk.tree.Tree):
            text = " ".join([word for word, tag in word.leaves()])
            return text
        else:
            continue
    return "Thing"


def wiki_classification(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    ne_chunked = nltk.ne_chunk(tagged)
    named_entities = extract_entities(ne_chunked)
    results = []
    for named_entity in named_entities:
        results.append([named_entity[0], wiki_definition(named_entity[0]), named_entity[2]])
    return results


def analyze(article):
    result = {
        'header': article['header'],
        'author': article['author'],
        'publish_date': article['publish_date']
    }
    body = [string for string in article['body'] if '\n' not in string]
    result['body'] = " ".join(body).replace('"', '“')
    result['pos_tagging'] = pos_tagging(result['body'])
    result['ner_entity_classification'] = ner_entity_classification(result['body'])
    result['ner_custom_pattern'] = ner_custom_pattern(result['body'])
    result['ner_hugging_face'] = ner_hugging_face(result['body'])
    result['wiki_classification'] = wiki_classification(result['body'])
    return result


########################################################################################################################
# INPUT / OUTPUT
########################################################################################################################
def open_data(input_file):
    f = open(input_file)
    data = json.load(f)
    f.close()
    return data


def pprint_pos_tagging(output, pos_t):
    output += f'    "POS_tagging": [\n'
    for i in range(len(pos_t)):
        output += f'      {json.dumps(pos_t[i])}'
        if i == len(pos_t) - 1:
            output += '\n'
        else:
            output += ',\n'
    output += f'    ],\n'
    return output


def pprint_ner_entity_classification(output, ner_e_c):
    output += f'    "NER_entity_classification": [\n'
    output += "      " + str(json.dumps(ner_e_c))[1:-1] + '\n'
    output += f'    ],\n'
    return output


def pprint_ner_custom_pattern(output, ner_c_p):
    output += f'    "NER_custom_pattern": [\n'
    output += "      " + str(json.dumps(ner_c_p))[1:-1] + '\n'
    output += f'    ],\n'
    return output


def pprint_ner_hugging_face(output, ner_h_f):
    output += f'    "NER_hugging_face_bert": [\n'
    output += "      " + str(json.dumps(ner_h_f))[1:-1] + '\n'
    output += f'    ],\n'
    return output


def pprint_wiki_classification(output, wiki_c):
    output += f'    "Wiki_classification": [\n'
    output += "      " + str(json.dumps(wiki_c))[1:-1] + '\n'
    output += f'    ]\n'
    return output


def pprint_result(results, output_path, std_out=False):
    output = ''
    output += '[\n'
    for i in range(len(results)):
        output += '  {\n'
        output += f'    "header": "{results[i]["header"]}",\n'
        output += f'    "author": "{results[i]["author"]}",\n'
        output += f'    "publish_date": "{results[i]["publish_date"]}",\n'
        output += f'    "body": "{results[i]["body"]}",\n'
        output = pprint_pos_tagging(output, results[i]["pos_tagging"])
        output = pprint_ner_entity_classification(output, results[i]["ner_entity_classification"])
        output = pprint_ner_custom_pattern (output, results[i]["ner_custom_pattern"])
        output = pprint_ner_hugging_face (output, results[i]["ner_hugging_face"])
        output = pprint_wiki_classification (output, results[i]["wiki_classification"])
        if i == len(results) - 1:
            output += '  }\n'
        else:
            output += '  },\n'
    output += ']\n'
    if std_out:
        print(output)
    with open(output_path, 'w') as output_file:
        print(output, file=output_file)


########################################################################################################################
# MAIN
########################################################################################################################
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    nltk.download(['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])
    results = []
    scraped_articles = open_data("data.json")
    for i in range(len(scraped_articles)):
        sys.stdout.write(f"\r{i+1} / {len(scraped_articles)}")
        sys.stdout.flush()
        results.append(analyze(scraped_articles[i]))
    print()
    pprint_result(results, '../results/result.json', True)
