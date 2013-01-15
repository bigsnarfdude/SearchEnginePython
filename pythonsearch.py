#! /usr/bin/env python

"""
pythonsearch.py
Python POC search engine lib

Usage

    import pythonsearch
    ps_instance = pythonsearch.Pythonsearch('path_to_data')

    # Index data
    ps_instance.index('email_1', {'text': "Peter,\n\nI'm going to need those TPS reports on my desk first thing tomorrow! And clean up your desk!\n\nLumbergh"})
    ps_instance.index('email_2', {'text': 'Everyone,\n\nM-m-m-m-my red stapler has gone missing. H-h-has a-an-anyone seen it?\n\nMilton'})
    ps_instance.index('email_3', {'text': "Peter,\n\nYeah, I'm going to need you to come in on Saturday. Don't forget those reports.\n\nLumbergh"})
    ps_instance.index('email_4', {'text': 'How do you feel about becoming Management?\n\nThe Bobs'})

    # Search
    ps_instance.search('Peter')
    ps_instance.search('tps report')

    # Documents are keys are field names
    # Values are the field's contents
    {
        "id": "document-1524",
        "text": "This is a blob of text. Nothing special about the text, just a typical document.",
        "created": "2012-02-18T20:19:00-0000",
    }

	# The (inverted) index itself (represented by the segment file bits), is also
	# essentially a dictionary. The difference is that the index is term-based, unlike
	# the field-based nature of the document:
    # Keys are terms
    # Values are document/position information
    index = {
        'blob': {
            'document-1524': [3],
        },
        'text': {
            'document-1524': [5, 10],
        },
    }

	# For this library, on disk, this is represented by a large number of small
	# segment files. You hash the term in question & take the first 6 chars of the
	# hash to determine what segment file it should be in. Those files are
	# maintained in alphabetical order. They look something like:

    blob\t{'document-1523': [3]}\n
    text\t{'document-1523': [5, 10]}\n

"""


import hashlib
import json
import math
import os
import re
import tempfile
# import NLTK stopwords here
# use stemmer from NLTK ml_intro


class PythonSearch(object):
    """
    Controls the indexing/searching of documents

    Usage:

        ps_instance = pythonsearch.Pythonsearch('path_to_data')
        ps_instance.index('email_1', {'text': "This is a blob of text to be indexed."})
        ps_instance.search('blob')

    """

    STOP_WORDS = set([
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by',
        'for', 'if', 'in', 'into', 'is', 'it',
        'no', 'not', 'of', 'on', 'or', 's', 'such',
        't', 'that', 'the', 'their', 'then', 'there', 'these',
        'they', 'this', 'to', 'was', 'will', 'with'
		])
    PUNCTUATION = re.compile('[~`!@#$%^&*()+={\[}\]|\\:;"\',<.>/?]')


    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.index_path = os.path.join(self.base_directory, 'index')
        self.docs_path = os.path.join(self.base_directory, 'documents')
        self.stats_path = os.path.join(self.base_directory, 'stats.json')
        self.setup()

    def setup(self):
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)

        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        if not os.path.exists(self.docs_path):
            os.makedirs(self.docs_path)

        return True

    def read_stats(self):
        if not os.path.exists(self.stats_path):
            return {
                'version': '.'.join([str(bit) for bit in __version__]),
                'total_docs': 0,
            }

        with open(self.stats_path, 'r') as stats_file:
            return json.load(stats_file)

    def write_stats(self, new_stats):
        with open(self.stats_path, 'w') as stats_file:
            json.dump(new_stats, stats_file)

        return True

    def increment_total_docs(self):
        current_stats = self.read_stats()
        current_stats.setdefault('total_docs', 0)
        current_stats['total_docs'] += 1
        self.write_stats(current_stats)

    def get_total_docs(self):
        current_stats = self.read_stats()
        return int(current_stats.get('total_docs', 0))

    def make_tokens(self, blob):
        blob = self.PUNCTUATION.sub(' ', blob)
        tokens = []
        for token in blob.split():
            token = token.lower().strip()

            if not token in self.STOP_WORDS:
                tokens.append(token)

        return tokens

    def make_ngrams(self, tokens, min_gram=3, max_gram=6):
        terms = {}

        for position, token in enumerate(tokens):
            for window_length in range(min_gram, min(max_gram + 1, len(token) + 1)):
                gram = token[:window_length]
                terms.setdefault(gram, [])

                if not position in terms[gram]:
                    terms[gram].append(position)

        return terms

    def hash_name(self, term, length=6):
        term = term.encode('ascii', errors='ignore')
        hashed = hashlib.md5(term).hexdigest()
        return hashed[:length]

    def make_segment_name(self, term):
        return os.path.join(self.index_path, "{0}.index".format(self.hash_name(term)))

    def parse_record(self, line):
        return line.rstrip().split('\t', 1)

    def make_record(self, term, term_info):
        return "{0}\t{1}\n".format(term, json.dumps(term_info, ensure_ascii=False))

    def update_term_info(self, orig_info, new_info):
        for doc_id, positions in new_info.items():
            if not doc_id in orig_info:
                orig_info[doc_id] = positions
            else:
                orig_positions = set(orig_info.get(doc_id, []))
                new_positions = set(positions)
                orig_positions.update(new_positions)
                orig_info[doc_id] = list(orig_positions)

        return orig_info

    def save_segment(self, term, term_info, update=False):
        seg_name = self.make_segment_name(term)
        new_seg_file = tempfile.NamedTemporaryFile(delete=False)
        written = False

        if not os.path.exists(seg_name):
            with open(seg_name, 'w') as seg_file:
                seg_file.write('')

        with open(seg_name, 'r') as seg_file:
            for line in seg_file:
                seg_term, seg_term_info = self.parse_record(line)

                if not written and seg_term > term:
                    new_line = self.make_record(term, term_info)
                    new_seg_file.write(new_line.encode('utf-8'))
                    written = True
                elif seg_term == term:
                    if not update:
                        line = self.make_record(term, term_info)
                    else:
                        new_info = self.update_term_info(json.loads(seg_term_info), term_info)
                        line = self.make_record(term, new_info)

                    written = True

                new_seg_file.write(line.encode('utf-8'))

            if not written:
                line = self.make_record(term, term_info)
                new_seg_file.write(line.encode('utf-8'))

        new_seg_file.close()
        try:
            os.rename(new_seg_file.name, seg_name)
        except OSError:
            os.remove(seg_name)
            os.rename(new_seg_file.name, seg_name)
        return True

    def load_segment(self, term):
        seg_name = self.make_segment_name(term)

        if not os.path.exists(seg_name):
            return {}

        with open(seg_name, 'r') as seg_file:
            for line in seg_file:
                seg_term, term_info = self.parse_record(line)

                if seg_term == term:
                    # Found it.
                    return json.loads(term_info)

        return {}

    def make_document_name(self, doc_id):
        return os.path.join(self.docs_path, self.hash_name(doc_id), "{0}.json".format(doc_id))

    def save_document(self, doc_id, document):
        doc_path = self.make_document_name(doc_id)
        base_path = os.path.dirname(doc_path)

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        with open(doc_path, 'w') as doc_file:
            doc_file.write(json.dumps(document, ensure_ascii=False))

        return True

    def load_document(self, doc_id):
        doc_path = self.make_document_name(doc_id)

        with open(doc_path, 'r') as doc_file:
            data = json.loads(doc_file.read())

        return data


    def index(self, doc_id, document):
        if not hasattr(document, 'items'):
            raise AttributeError('You must provide `index` with a document in the form of a dictionary.')

        if not 'text' in document:
            raise KeyError('You must provide `index` with a document with a `text` field in it.')

        doc_id = str(doc_id)
        self.save_document(doc_id, document)

        tokens = self.make_tokens(document.get('text', ''))
        terms = self.make_ngrams(tokens)

        for term, positions in terms.items():
            self.save_segment(term, {doc_id: positions}, update=True)

        self.increment_total_docs()
        return True

    def parse_query(self, query):
        tokens = self.make_tokens(query)
        return self.make_ngrams(tokens)

    def collect_results(self, terms):
        per_term_docs = {}
        per_doc_counts = {}
        for term in terms:
            term_matches = self.load_segment(term)
            per_term_docs.setdefault(term, 0)
            per_term_docs[term] += len(term_matches.keys())
            for doc_id, positions in term_matches.items():
                per_doc_counts.setdefault(doc_id, {})
                per_doc_counts[doc_id].setdefault(term, 0)
                per_doc_counts[doc_id][term] += len(positions)

        return per_term_docs, per_doc_counts

    def bm25_relevance(self, terms, matches, current_doc, total_docs, b=0, k=1.2):
        score = b
        for term in terms:
            idf = math.log((total_docs - matches[term] + 1.0) / matches[term]) / math.log(1.0 + total_docs)
            score = score + current_doc.get(term, 0) * idf / (current_doc.get(term, 0) + k)

        return 0.5 + score / (2 * len(terms))

    def search(self, query, offset=0, limit=20):
        results = {
            'total_hits': 0,
            'results': []
        }

        if not len(query):
            return results

        total_docs = self.get_total_docs()

        if total_docs == 0:
            return results

        terms = self.parse_query(query)
        per_term_docs, per_doc_counts = self.collect_results(terms)
        scored_results = []
        final_results = []

        for doc_id, current_doc in per_doc_counts.items():
            scored_results.append({
                'id': doc_id,
                'score': self.bm25_relevance(terms, per_term_docs, current_doc, total_docs),
            })

        sorted_results = sorted(scored_results, key=lambda res: res['score'], reverse=True)
        results['total_hits'] = len(sorted_results)

        sliced_results = sorted_results[offset:offset + limit]

        for res in sliced_results:
            doc_dict = self.load_document(res['id'])
            doc_dict.update(res)
            results['results'].append(doc_dict)

        return results
